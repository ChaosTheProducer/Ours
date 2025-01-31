from functools import partial

import torch
import torch.nn as nn

from einops import rearrange

from timm.models.vision_transformer import Block
from .patch_embed import PatchEmbed3D

#from patch_embed import PatchEmbed3D


class MAE3D(nn.Module):
    """  3D Masked Autoencoder with ViT Backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=16,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # 3D-MAE encoder specifics
        self.patch_embed = PatchEmbed3D(img_size, patch_size, in_chans, embed_dim)
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim

        # Separable encoder positional embeddings

        self.pos_embed_spatial = nn.Parameter(
            torch.zeros(1, self.patch_embed.grid_size[0] * self.patch_embed.grid_size[1] * self.patch_embed.grid_size[2], embed_dim))

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)

        # 3D-MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # Separable decoder positional embeddings

        self.decoder_pos_embed_spatial = nn.Parameter(
            torch.zeros(1,self.patch_embed.grid_size[0] * self.patch_embed.grid_size[1] * self.patch_embed.grid_size[2], decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 3 * in_chans, bias=True)  # decoder to patch

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.pos_embed_spatial, std=.02)

        torch.nn.init.normal_(self.mask_token, std=.02)

        torch.nn.init.normal_(self.decoder_pos_embed_spatial, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)



    def patchify3D(self, imgs):
        """
        imgs: (N, 3, H, W, D)
        x: (N, L, patch_size**3 * c )
        """
        x = rearrange(imgs, 'b c (h p0) (w p1) (d p2) -> b (h w d) (p0 p1 p2) c',
                      p0=self.patch_embed.patch_size[0],
                      p1=self.patch_embed.patch_size[1],
                      p2=self.patch_embed.patch_size[2])
        x = rearrange(x, 'b n p c -> b n (p c)')
        return x

    def unpatchify3D(self, x):
        """
        x: (N, L, patch_size**3 * c)
        imgs: (N, 3, H, W, D)
        """
        x = rearrange(x, 'b (h w d) (p0 p1 p2 c) -> b c (h p0) (w p1) (d p2)',
                      p0 = self.patch_embed.patch_size[0],
                      p1 = self.patch_embed.patch_size[1],
                      p2 = self.patch_embed.patch_size[2],
                      c  = self.in_chans,
                      h  = self.patch_embed.grid_size[0],
                      w  = self.patch_embed.grid_size[1],
                      d  = self.patch_embed.grid_size[2])
        return x

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        #随机化一个0到1的噪声
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        # 然后为噪声从小到大排序，保存索引位置，这样就可以对patch_embedding进行打乱
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        # 再重新排序一下， 就可以得到原有数据排列的索引
        ids_restore = torch.argsort(ids_shuffle, dim=1)


        # keep the first subset
        # 保存噪声小的索引位置，这样就可以对原有数据的patch排列方式进行打乱
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        """
        x_masked：只包含保留的token的tensor，形状为[N, len_keep, D]
        mask：二进制掩码，表示每个位置是否被掩码，形状为[N, L]
        ids_restore：用于将掩码恢复到原始顺序的索引
        """

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        x = self.patch_embed(x)

        # add pos embed
        x = x + self.pos_embed_spatial

        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):

        x = self.decoder_embed(x)

        # 2. 生成 mask token，替代被遮挡的部分
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)  # 形状: [N, M, decoder_embed_dim]
        x_ = torch.cat([x, mask_tokens], dim=1)  # 将 mask token 加入到解码器输入中，形状变为 [N, L + M, decoder_embed_dim]
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # 3. 添加位置编码
        x = x + self.decoder_pos_embed_spatial  # shape: [N, L + M, decoder_embed_dim]

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        return x  # 返回恢复后的补丁

    # MSE loss
    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W, D]
        pred: (N, L, patch_size**3 *c)
        mask: [N, L], 0 is keep, 1 is remove
        """

        target = self.patchify3D(imgs)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    # original mask ratio is 0.90
    #def forward(self, imgs, mask_ratio=0.75):
    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # (N, L, patch_size**3 *c)
        loss = self.forward_loss(imgs, pred, mask)
        pred = self.unpatchify3D(pred)
        return loss, pred, mask

"""
if __name__ == '__main__':
    #x = torch.rand(1, 3, 32, 32, 32)
    #model = MAE3D(img_size=32,patch_size=8,in_chans=3,embed_dim=64,depth=2,num_heads=4,
                  #decoder_embed_dim=128, decoder_depth=4,decoder_num_heads=4,mlp_ratio=4.)
    x = torch.rand(1, 16, 32, 32, 8)
    model = MAE3D(img_size=32,patch_size=8,in_chans=16,embed_dim=64,depth=2,num_heads=4,
                  decoder_embed_dim=128, decoder_depth=4,decoder_num_heads=4,mlp_ratio=4.)
    x = model(x)
    print(x[1].shape)
"""
