"""

ViT transformer

Based primarily on a video tutorial from Vision Transformer
at https://www.youtube.com/watch?v=ovB0ddFtzzA&ab_channel=mildlyoverfitted

and 

Official code PyTorch implementation from CDTrans paper:
https://github.com/CDTrans/CDTrans

"""
import math
import copy
from functools import partial
from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable, Union, Dict


class PatchEmbed(nn.Module):
    """
    Split image into patches and then embed them.

    Parameters
    ----------
    img_size : int (square)
    patch_size : int (square)
    in_chans : int
    embed_dim : int

    Atttributes:
    -----------
    n_patches : int
    proj : nn.Conv2d

    """
    def __init__(self, img_size, patch_size, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        # self.patch_size = patch_size
        self.n_patches = (img_size[1] // patch_size) * (img_size[2] // patch_size)
        
        self.proj = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True
        )


    def forward(self, x):
        """
        Input
        ------
        x : Shape (n_samples, in_chans, img_size, img_size)

        Returns:
        --------
        Shape (n_samples, n_patches, embed_dims)
        """
        x = self.proj(x) # out: (n_samples, embed_dim, n_patches[0], n_patches[1])
        x = x.flatten(2) # out: (n_samples, embed_dim, n_patches)
        x = x.transpose(1, 2) # out: (n_samples, n_patches, embed_dim)

        return x


class Attention(nn.Module):
    """
    Attention mechanism

    Parameters
    -----------
    dim : int (dim per token features)
    n_heads : int
    qkv_bias : bool
    attn_p : float (Dropout applied to q, k, v)
    proj_p : float (Dropout applied to output tensor)

    Attributes
    ----------
    scale : float
    qkv : nn.Linear
    proj : nn.Linear
    attn_drop, proj_drop : nn.Dropout
    
    """
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)
    
    def forward(self, x):
        """
        Input
        ------
        x : Shape (n_samples, n_patches + 1, dim)

        Returns:
        -------
        Shape (n_samples, n_patches + 1, dim)

        """
        n_samples, n_tokens, dim =  x.shape

        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x) # (n_samples, n_patches + 1, 3 * dim)

        qkv = qkv.reshape(
            n_samples, n_tokens, 3, self.n_heads, self.head_dim
        ) # (n_samples, n_patches + 1, 3, n_heads, head_dim)
        qkv = qkv.permute(
            2, 0, 3, 1, 4
        ) # (3, n_samples, n_heads, n_patches + 1, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2] # each with (n_samples, n_heads, n_patches + 1, head_dim)

        k_t = k.transpose(-2, -1) # (n_samples, n_heads, head_dim, n_patches + 1)
        dp = (
            q @ k_t
        ) * self.scale # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = dp.softmax(dim=-1) # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v # (n_samples, n_heads, n_patches + 1, head_dim)
        weighted_avg = weighted_avg.transpose(
            1, 2
        ) # (n_samples, n_patches + 1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2) # (n_samples, n_patches + 1, dim)
        x = self.proj(weighted_avg)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    """
    Multilayer Perceptron

    Parameters
    ----------
    in_features : int
    hidden_features : int
    out_features : int
    p : float

    Attributes
    ---------
    fc1 : nn.Linear
    act : nn.GELU
    fc2 : nn.Linear
    drop : nn.Dropout
    """
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        """
        Input
        ------
        Shape (n_samples, n_patches + 1, in_features)

        Returns:
        ---------
        Shape (n_samples, n_patches + 1, out_features)
        """
        x = self.fc1(
            x
            ) # (n_samples, n_patches + 1, hidden_features)
        x = self.act(x) # (n_samples, n_patches + 1, hidden_features)
        x = self.drop(x) # (n_samples, n_patches + 1, hidden_features)
        x = self.fc2(x) # (n_samples, n_patches + 1, out_features)
        x = self.drop(x) # (n_samples, n_patches + 1, out_features)

        return x

class Block(nn.Module):
    """
    Transformer block

    Parameters
    ----------
    dim : int
    n_heads : int
    mlp_ratio : float
    qkv_bias : book
    p, attn_p : float

    Attributes
    ----------
    norm1, norm2 : LayerNorm
    attn : Attention
    mlp : MLP
    """
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_p=attn_p,
            proj_p=p
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=hidden_features,
            out_features=dim
        )

    def forward(self, x):
        """
        Input
        ------
        Shape (n_samples, n_patches + 1, dim)

        Returns:
        ---------
        Shape (n_samples, n_patches + 1, dim)
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x


class Vision_Transformer(nn.Module):
    """
    3D Vision Transformer

    Parameters
    -----------
    img_size : int
    patch_size : int
    in_chans : int
    n_classes : int
    embed_dim : int
    depth : int
    n_heads : int
    mlp_ratio : float
    qkv_bias : book
    p, attn_p : float

    Attributes:
    -----------
    patch_embed : PatchEmbed
    cls_token : nn.Parameter
    pos_emb : nn.Parameter
    pos_drop : nn.Dropout
    blocks : nn.ModuleList
    norm : nn.LayerNorm
    """
    def __init__(self, 
                img_size=(3,32,32), 
                patch_size=16, 
                in_chans=3, 
                n_classes=1000, 
                embed_dim=768, 
                depth=12, 
                n_heads=12, 
                mlp_ratio=4., 
                qkv_bias=True, 
                p=0., 
                attn_p=0.,
                weight_init=''
                ):
        super().__init__()

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim
        )

        self.cls_token = nn.Parameter(torch.rand(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.rand(1, 1 + self.patch_embed.n_patches, embed_dim)
        )
        
        self.pos_drop = nn.Dropout(p=p)

        self.blocks = nn.Sequential(*[
                Block(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p
                )
                for __ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # self.prehead_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

        # self.bottleneck = nn.BatchNorm1d(embed_dim)
        # self.bottleneck.bias.requires_grad_(False)
        # self.bottleneck.apply(weights_init_kaiming)

        # trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.pos_embed, std=.02)

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)
        print("Model weights initialized")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        Input
        -----
        Shape (n_samples, in_chans, img_size, img_size)

        Returns:
        --------
        Shape (n_samples, n_classes)
        
        """
        n_samples = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(
            n_samples, -1, -1
        ) # (n_samples, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1) # (n_samples, 1 + n_patches, embed_dim)
        x = x + self.pos_embed# (n_samples, 1 + n_patches, embed_dim)
        x = self.pos_drop(x)


        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)

        cls_token_final = x[:, 0] # just the CLS token
        # cls_token_final = self.bottleneck(cls_token_final)
        # cls_token_final = self.prehead_norm(cls_token_final)
        x = self.head(cls_token_final)

        return x
    
    def save(self,checkpoint):
        torch.save(self.state_dict(), checkpoint)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        print("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def get_init_weights_vit(mode='jax', head_bias: float = 0.):
    if 'jax' in mode:
        return partial(init_weights_vit_jax, head_bias=head_bias)
    elif 'moco' in mode:
        return init_weights_vit_moco
    else:
        return init_weights_vit_timm

def init_weights_vit_timm(module: nn.Module, name: str = ''):
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    ## type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)  

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def named_apply(fn: Callable, module: nn.Module, name='', depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = '.'.join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module



