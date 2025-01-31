from torch import nn as nn
from torch import _assert
import torch

class PatchEmbed3D(nn.Module):
    """ 3D Image to Patch Embedding """
    def __init__(self, img_size=224, patch_size=16, in_chans=16, embed_dim=768,
                 norm_layer=None, flatten=True):
        super().__init__()
        img_size = (img_size, img_size, 8)  # cardiac
        #img_size = (img_size, img_size, 32)  # lung
        patch_size = (patch_size, patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])
        self.flatten = flatten

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W, D = x.shape
        _assert(H == self.img_size[0], f"Input height ({H}) doesn't match model ({self.img_size[0]}).")
        _assert(W == self.img_size[1], f"Input width ({W}) doesn't match model ({self.img_size[1]}).")
        _assert(D == self.img_size[2], f"Input depth ({D}) doesn't match model ({self.img_size[2]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2) # BCHWD -> BNC
        x = self.norm(x)
        return x

"""
if __name__ == '__main__':
    x = torch.rand(1, 3, 32, 32, 32)
    model = PatchEmbed3D(img_size=32,patch_size=16,in_chans=3,embed_dim=32)
    x = model(x)
    print(x.shape)
"""