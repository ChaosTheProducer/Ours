import torch
import torch.nn as nn

from models.mae3d.model_3d_mae import MAE3D  # 使用绝对导入

class MaskPredictor(nn.Module):
    def __init__(self, img_size=32, patch_size=8, in_chans=16, embed_dim=64, depth=2, num_heads=4,
                 decoder_embed_dim=64, decoder_depth=2, decoder_num_heads=4, mlp_ratio=4., norm_layer=nn.LayerNorm):
        super(MaskPredictor, self).__init__()
        
        # 使用完整的 MAE3D 模型作为主模型
        self.mae_model = MAE3D(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=in_chans, 
            embed_dim=embed_dim, 
            depth=depth, 
            num_heads=num_heads, 
            decoder_embed_dim=decoder_embed_dim, 
            decoder_depth=decoder_depth, 
            decoder_num_heads=decoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            norm_layer=norm_layer
        )

    def forward(self, x, mask_ratio=0.75):
        pred, mask, ids_restore = self.mae_model(x, mask_ratio=mask_ratio)
        return pred, mask, ids_restore

# 示例用法
if __name__ == '__main__':
    x = torch.rand(1, 16, 32, 32, 8)
    model = MaskPredictor(img_size=(32, 32, 8), patch_size=8, in_chans=16, embed_dim=64, depth=2, num_heads=4,
                          decoder_embed_dim=64, decoder_depth=2, decoder_num_heads=4, mlp_ratio=4., norm_layer=nn.LayerNorm)
    pred, mask, ids_restore = model(x)
    print(f"Prediction Shape: {pred.shape}")  # 输出: torch.Size([1, 16, 32, 32, 8])
    print(f"Mask Shape: {mask.shape}")        # 输出: torch.Size([1, 4]) 或者 (batch_size, num_patches)
    print(f"IDs Restore Shape: {ids_restore.shape}")  # 输出: torch.Size([1, 4]) 或者 (batch_size, num_patches)



