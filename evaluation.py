import os
import random
from argparse import ArgumentParser
from math import log10, sqrt

import losses
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn as nn
from utils import datasets, utils
from lpips import LPIPS
from models.UNet.model import Unet3D, Unet3D_multi
from models.VoxelMorph.model import VoxelMorph
from natsort import natsorted
from torch.utils.data import DataLoader
from models.feature_extract.model import FeatureExtract
#from models.AuxiliaryF.g import RotationPredictor
from models.mae3d.model_3d_mae import MAE3D

import matplotlib.pyplot as plt
import imageio
import imutils

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True


def evaluate(original, generated):
    max_pixel = 1.0

    #### PSNR
    mse = torch.mean((original - generated) ** 2)
    psnr = 20 * log10(max_pixel / sqrt(mse))

    ### NCC
    criterion_ncc = losses.NCC()
    ncc = criterion_ncc(original, generated)

    #### SSIM
    criterion_ssim = losses.SSIM3D()
    ssim = 1 - criterion_ssim(original, generated)

    #### NMSE
    nmse = mse / torch.mean(original**2)

    #### LPIPS
    lpips = utils.AverageMeter()
    new_original_image, new_generated_image = [], []
    for i in range(generated.shape[2]):
        new_original_image.append((original[0, 0, i, :, :] * 2 - 1).repeat(3, 1, 1))
        new_generated_image.append((generated[0, 0, i, :, :] * 2 - 1).repeat(3, 1, 1))
    for i in range(generated.shape[3]):
        new_original_image.append((original[0, 0, :, i, :] * 2 - 1).repeat(3, 1, 1))
        new_generated_image.append((generated[0, 0, :, i, :] * 2 - 1).repeat(3, 1, 1))
    original_image = torch.stack(new_original_image, dim=0).cuda()
    generated_image = torch.stack(new_generated_image, dim=0).cuda()
    lpips_ = loss_fn_alex(original_image, generated_image)
    for l in lpips_:
        lpips.update(l.item())

    new_original_image, new_generated_image = [], []
    for i in range(generated.shape[4]):
        new_original_image.append((original[0, 0, :, :, i] * 2 - 1).repeat(3, 1, 1))
        new_generated_image.append((generated[0, 0, :, :, i] * 2 - 1).repeat(3, 1, 1))
    original_image = torch.stack(new_original_image, dim=0).cuda()
    generated_image = torch.stack(new_generated_image, dim=0).cuda()
    lpips_ = loss_fn_alex(original_image, generated_image)
    for l in lpips_:
        lpips.update(l.item())

    return psnr, -1 * ncc.item(), ssim.item(), nmse.item(), lpips.avg

# New rotate function
def rotate_xy_plane(tensor, degree):
    """
    对 5D Tensor 的 x-y 平面进行旋转（绕 z 轴）。
    Args:
        tensor (torch.Tensor): 输入 Tensor，形状为 [B, C, H, W, D]。
        degree (int): 旋转角度，支持 90, 180, 270。
    Returns:
        torch.Tensor: 旋转后的 Tensor，形状与输入相同。
    """
    if degree not in [0, 90, 180, 270]:
        raise ValueError("只支持 90°, 180°, 270° 的旋转")
    
    # 复制输入 Tensor，以免影响原数据
    rotated_tensor = tensor.clone()

    if degree == 90:
        # 90° 旋转：transpose + flip
        rotated_tensor = rotated_tensor.permute(0, 1, 3, 2, 4).flip(3)
    elif degree == 180:
        # 180° 旋转：双 flip
        rotated_tensor = rotated_tensor.flip(2).flip(3)
    elif degree == 270:
        # 270° 旋转：transpose + flip
        rotated_tensor = rotated_tensor.permute(0, 1, 3, 2, 4).flip(2)
    elif degree == 0:
        rotated_tensor = rotated_tensor

    return rotated_tensor


def main(args):
    set_seed(args.seed)

    save_dir = f"experiments/{args.dataset}"

    if args.dataset == "cardiac":
        img_size = (128, 128, 32)
        split = 90 if args.split is None else args.split
    elif args.dataset == "lung":
        img_size = (128, 128, 128)
        split = 68 if args.split is None else args.split

    """
    Initialize model
    """
    flow_model = VoxelMorph(img_size).cuda()
    if args.feature_extract:
        refinement_model = Unet3D_multi(img_size).cuda()
    else:
        refinement_model = Unet3D(img_size).cuda()

    if args.feature_extract:
        feature_model = FeatureExtract().cuda()

    # Bigger mae - Cardiac使用
    if args.dataset == "cardiac":
        mae3d_model = MAE3D(img_size=32,patch_size=8,in_chans=16,embed_dim=256,depth=6,num_heads=8,
                  decoder_embed_dim=256, decoder_depth=6,decoder_num_heads=8,mlp_ratio=4.).cuda()
    elif args.dataset == "lung":
    # 中等大小-lung使用
        mae3d_model = MAE3D(img_size=32,patch_size=8,in_chans=16,embed_dim=256,depth=6,num_heads=8,
                  decoder_embed_dim=256, decoder_depth=6,decoder_num_heads=8,mlp_ratio=4.).cuda()
    

    """
    if args.dataset == "cardiac":
        rotation_predictor = RotationPredictor(input_dim=16 * (32 // 4) * (128 // 4) * (128 // 4)).cuda()
    elif args.dataset == "lung":
        rotation_predictor = RotationPredictor(input_dim=16 * (32 // 4) * (128 // 4) * (128 // 4)).cuda()

    """
    optimizer = torch.optim.Adam(
        #list(feature_model.parameters()) + list(rotation_predictor.parameters()), 
        list(feature_model.parameters()), #+ list(mae3d_model.parameters()),
        lr=1e-4
    )

    optimizer_mae = torch.optim.AdamW(
            list(mae3d_model.parameters()),
            #lr=1e-4,
            lr=2e-4,
            eps=1e-8,
            weight_decay=0.05,
        )


    criterion = nn.CrossEntropyLoss()

    """Load checkpoint"""
    best_model_path = natsorted(os.listdir(save_dir))[args.model_idx]
    best_model = torch.load(os.path.join(save_dir, best_model_path))
    print("Model: {} loaded!".format(best_model_path))
    flow_model.load_state_dict(best_model["flow_model_state_dict"])
    refinement_model.load_state_dict(best_model["model_state_dict"])
    mae3d_model.load_state_dict(best_model["mae3d_model_state_dict"])  # load best 3d mae model

    if args.feature_extract:
        print("Use Feature Extract!")
        feature_model.load_state_dict(best_model["feature_model_state_dict"])

    f0 = {
        "feature_model": feature_model.state_dict(),
        "3d_mae": mae3d_model.state_dict(),
        "refinement_model":refinement_model.state_dict(),
    }

    """
    Initialize spatial transformation function
    """
    reg_model = utils.register_model(img_size, "nearest")
    reg_model.cuda()
    reg_model_bilin = utils.register_model(img_size, "bilinear")
    reg_model_bilin.cuda()

    """
    Initialize training
    """
    if args.dataset == "cardiac":
        data_dir = os.path.join("dataset", "ACDC", "database", "training")
        val_set = datasets.ACDCHeartDataset(data_dir, phase="test", split=split)
    elif args.dataset == "lung":
        data_dir = os.path.join("dataset", "4D-Lung-Preprocessed")
        val_set = datasets.LungDataset(data_dir, phase="test", split=split)
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        drop_last=True,
    )

    global loss_fn_alex
    loss_fn_alex = LPIPS(net="alex").cuda()

    psnr_log = utils.AverageMeter()
    nmse_log = utils.AverageMeter()
    ncc_log = utils.AverageMeter()
    lpips_log = utils.AverageMeter()
    ssim_log = utils.AverageMeter()


    """
    Test Time Training
    """

    loss_aux_meter = utils.AverageMeter()  # 记录平均损失
    mask_ratio = 0.75
    if args.ttt_mode == 'naive':  # Naive TTT
        """
        Naive TTT - Update model after all mini-batches (Multiple epochs for all data)
        """
        # 加载初始权重一次
        feature_model.load_state_dict(f0["feature_model"])
        mae3d_model.load_state_dict(f0["3d_mae"])
        refinement_model.load_state_dict(f0["refinement_model"])

        feature_model.train()
        mae3d_model.train()
        refinement_model.train()

        for epoch in range(50):  # 多个 epoch
            loss_aux_meter.reset()
            for data_idx, data in enumerate(val_loader):  # 遍历所有 mini-batch
                optimizer_mae.zero_grad()

                # 数据处理
                data = [t.cuda() for t in data]
                i0 = data[0]
                true_rotation = torch.tensor(
                    [random.choice([0, 90, 180, 270]) for _ in range(i0.size(0))], 
                    device=i0.device
                )
                rotated_tensor = rotate_xy_plane(i0, true_rotation)

                # 提取特征
                features_clone = feature_model(rotated_tensor.clone().detach())[-1]

                # Forward pass
                loss_aux, pred, mask = mae3d_model(features_clone, mask_ratio)

                # Backward pass and optimization
                loss_aux.backward()
                optimizer_mae.step()

                # 记录损失
                loss_aux_meter.update(loss_aux.item(), i0.size(0))

            print(f"Epoch {epoch+1}/50, Loss_aux(MAE): {loss_aux_meter.avg}")

        print(f"Finished Naive Test Time Training for 50 epochs.")

    elif args.ttt_mode == 'online':  # Online TTT
        """
        Online TTT - Update model after each mini-batch independently (weights reset to θ0 for each mini-batch)
        """
        for epoch in range(50):
            for data_idx, data in enumerate(val_loader):
                # 重置模型权重到初始状态
                feature_model.load_state_dict(f0["feature_model"])
                mae3d_model.load_state_dict(f0["3d_mae"])
                refinement_model.load_state_dict(f0["refinement_model"])

                feature_model.train()
                mae3d_model.train()
                refinement_model.train()

                optimizer_mae.zero_grad()

                # 数据处理
                data = [t.cuda() for t in data]
                i0 = data[0]
                true_rotation = torch.tensor(
                    [random.choice([0, 90, 180, 270]) for _ in range(i0.size(0))],
                    device=i0.device
                )
                rotated_tensor = rotate_xy_plane(i0, true_rotation)

                # 提取特征
                features_clone = feature_model(rotated_tensor.clone().detach())[-1]

                # Forward pass
                loss_aux, pred, mask = mae3d_model(features_clone, mask_ratio)

                # Backward pass and optimization
                loss_aux.backward()
                optimizer_mae.step()

                # 记录损失
                loss_aux_meter.update(loss_aux.item(), i0.size(0))

                print(f"Batch {data_idx+1}, Loss_aux(MAE): {loss_aux.item()}")

        print(f"Finished Online Test Time Training for {data_idx+1} batches.")

    elif args.ttt_mode == 'mini_batch':  # Mini-Batch TTT
        """
        Mini-batch TTT - Update model after each mini-batch (Iterative updates)
        """
        # 加载初始权重一次
        feature_model.load_state_dict(f0["feature_model"])
        mae3d_model.load_state_dict(f0["3d_mae"])
        refinement_model.load_state_dict(f0["refinement_model"])

        feature_model.train()
        mae3d_model.train()
        refinement_model.train()

        for epoch in range(50):  # 多个 epoch
            loss_aux_meter.reset()
            for data_idx, data in enumerate(val_loader):  # 遍历所有 mini-batch
                optimizer_mae.zero_grad()

                # 数据处理
                data = [t.cuda() for t in data]
                i0 = data[0]
                true_rotation = torch.tensor(
                    [random.choice([0, 90, 180, 270]) for _ in range(i0.size(0))],
                    device=i0.device
                )
                rotated_tensor = rotate_xy_plane(i0, true_rotation)

                # 提取特征
                features_clone = feature_model(rotated_tensor.clone().detach())[-1]

                # Forward pass
                loss_aux, pred, mask = mae3d_model(features_clone, mask_ratio)

                # Backward pass and optimization
                loss_aux.backward()
                optimizer_mae.step()

                # 记录损失
                loss_aux_meter.update(loss_aux.item(), i0.size(0))

                print(f"Epoch {epoch+1}/50, Batch {data_idx+1}, Loss_aux(MAE): {loss_aux_meter.avg}")

        print(f"Finished Mini-batch Test Time Training for 50 epochs.")

    
    """
    原始Evaluation起点
    """
    for data_idx, data in enumerate(val_loader):
        
        #interface
        feature_model.eval()
        mae3d_model.eval()
        refinement_model.eval()

        data = [t.cuda() for t in data]
        i0 = data[0]
        i1 = data[1]
        idx0, idx1 = data[2], data[3]
        video = data[4]

        i0_i1 = torch.cat((i0, i1), dim=1)
        _, _, flow_0_1, flow_1_0 = flow_model(i0_i1)

        for i in range(idx0 + 1, idx1):
            alpha = (i - idx0) / (idx1 - idx0)

            flow_0_alpha = flow_0_1 * alpha
            flow_1_alpha = flow_1_0 * (1 - alpha)

            i_0_alpha = reg_model_bilin([i0, flow_0_alpha.float()])
            i_1_alpha = reg_model_bilin([i1, flow_1_alpha.float()])

            i_alpha_combined = (1 - alpha) * i_0_alpha + alpha * i_1_alpha

            if args.feature_extract:
                x_feat_i0_list = feature_model(i0)
                #print(feature_model(i0))
                x_feat_i1_list = feature_model(i1)
                x_feat_i0_alpha_list, x_feat_i1_alpha_list = [], []

                for idx in range(len(x_feat_i0_list)):
                    reg_model_feat = utils.register_model(
                        tuple([x // (2**idx) for x in img_size])
                    )
                    x_feat_i0_alpha_list.append(
                        reg_model_feat(
                            [
                                x_feat_i0_list[idx],
                                F.interpolate(
                                    flow_0_alpha * (0.5 ** (idx)),
                                    scale_factor=0.5 ** (idx),
                                ).float(),
                            ]
                        )
                    )
                    x_feat_i1_alpha_list.append(
                        reg_model_feat(
                            [
                                x_feat_i1_list[idx],
                                F.interpolate(
                                    flow_1_alpha * (0.5 ** (idx)),
                                    scale_factor=0.5 ** (idx),
                                ).float(),
                            ]
                        )
                    )

                i_alpha_out_diff = refinement_model(
                    i_alpha_combined, x_feat_i0_alpha_list, x_feat_i1_alpha_list
                )

            else:
                i_alpha_out_diff = refinement_model(i_alpha_combined)

            i_alpha = i_alpha_combined + i_alpha_out_diff

            gt_alpha = video[..., i]


            """
            Generate figs
            """

            # 创建保存目录
            output_dir = os.path.join("Evaluation_output", args.dataset, f"batch_{data_idx}")
            os.makedirs(output_dir, exist_ok=True)

            #   获取完整的 3D 数据
            real_volume = gt_alpha[0, 0].detach().cpu().numpy()  
            generated_volume = i_alpha[0, 0].detach().cpu().numpy()  

            # 选择切片索引
            if args.dataset == "cardiac":
                slice_idx = real_volume.shape[2] // 2
            elif args.dataset == "lung":
                slice_idx = real_volume.shape[2] // 2
            else:
                raise ValueError(f"Unsupported dataset: {args.dataset}")

            # 选择切片
            if args.dataset == "cardiac":
                real_slice = real_volume[:, :, slice_idx]  # 短轴切片  # 冠状切片
                generated_slice = generated_volume[:, :, slice_idx]  # 冠状切片
            elif args.dataset == "lung":
                real_slice = real_volume[:, slice_idx, :]  # 矢状切片
                generated_slice = generated_volume[:, slice_idx, :]  # 矢状切片

            # 计算差异图
            difference = np.abs(real_slice - generated_slice)

            # 定义一个固定的画布大小，确保图片不变形
            fixed_figsize = (6, 6)  # 可以根据需要调整

            # 保存真实图片（Ground Truth）
            plt.figure(figsize=fixed_figsize)
            plt.imshow(real_slice, cmap='gray', aspect='equal')  # 使用 aspect='equal' 保持比例
            plt.axis('off')  # 关闭坐标轴
            output_path = os.path.join(output_dir, f"frame_{i}_ground_truth.png")
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close()

            # 保存插值生成的图片（Interpolated）
            plt.figure(figsize=fixed_figsize)
            plt.imshow(generated_slice, cmap='gray', aspect='equal')  # 使用 aspect='equal' 保持比例
            plt.axis('off')  # 关闭坐标轴
            output_path = os.path.join(output_dir, f"frame_{i}_interpolated.png")
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close()

            # 保存差异图
            plt.figure(figsize=fixed_figsize)
            plt.imshow(difference, cmap='hot', aspect='equal')  # 使用 aspect='equal' 保持比例
            plt.axis('off')  # 关闭坐标轴
            output_path = os.path.join(output_dir, f"frame_{i}_difference.png")
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close()

            # 打印保存路径
            print(f"真实图片已保存至: {os.path.join(output_dir, f'frame_{i}_ground_truth.png')}")
            print(f"插值图片已保存至: {os.path.join(output_dir, f'frame_{i}_interpolated.png')}")
            print(f"差异图已保存至: {os.path.join(output_dir, f'frame_{i}_difference.png')}")
            

            """
            Call evaluation
            """
            psnr, ncc, ssim, nmse, lpips = evaluate(gt_alpha, i_alpha)

            psnr_log.update(psnr)
            nmse_log.update(nmse)
            ncc_log.update(ncc)
            lpips_log.update(lpips)
            ssim_log.update(ssim)

    print(
        "AVG\tPSNR: {:2.2f}, NCC: {:.3f}, SSIM: {:.3f}, NMSE: {:.3f}, LPIPS: {:.3f}".format(
            psnr_log.avg,
            ncc_log.avg,
            ssim_log.avg,
            nmse_log.avg * 100,
            lpips_log.avg * 100,
        )
    )
    print(
        "STDERR\tPSNR: {:.3f}, NCC: {:.3f}, SSIM: {:.3f}, NMSE: {:.3f}, LPIPS: {:.3f}\n".format(
            psnr_log.stderr,
            ncc_log.stderr,
            ssim_log.stderr,
            nmse_log.stderr * 100,
            lpips_log.stderr * 100,
        )
    )


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=str, default=None)

    parser.add_argument(
        "--dataset", type=str, default="cardiac", choices=["cardiac", "lung"]
    )
    parser.add_argument("--model_idx", type=int, default=-1)
    parser.add_argument("--split", type=int, default=None)
    parser.add_argument("--feature_extract", action="store_true", default=True)
    parser.add_argument("--ttt_mode", type=str, choices=["naive", "online", "mini_batch"], default="naive", help="Choose the Test Time Training mode: 'naive', 'online', or 'mini_batch'")

    args = parser.parse_args()

    """
    GPU configuration
    """
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print("Number of GPU: " + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print("     GPU #" + str(GPU_idx) + ": " + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print("Currently using: " + torch.cuda.get_device_name(GPU_iden))
    print("If the GPU is available? " + str(GPU_avai))

    main(args)
