"""

Implementation from https://www.youtube.com/watch?v=ovB0ddFtzzA&ab_channel=mildlyoverfitted

"""


import torch
import torch.nn as nn

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
    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
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
        x = self.proj(x) # (n_samples, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(2) # (n_samples, embed_dim, n_patches)
        x = x.transpose(1, 2) # (n_samples, n_patches, embed_dim)

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
    Vision Transformer

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
                img_size=384, 
                patch_size=16, 
                in_chans=3, 
                n_classes=1000, 
                embed_dim=768, 
                depth=12, 
                n_heads=12, 
                mlp_ratio=4., 
                qkv_bias=True, 
                p=0., 
                attn_p=0.
                ):
        super().__init__()

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim)
        )
        
        self.pos_drop = nn.Dropout(p=p)

        self.blocks = nn.ModuleList(
            [
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
        self.head = nn.Linear(embed_dim, n_classes)

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
        x = x + self.pos_embed # (n_samples, 1 + n_patches, embed_dim)
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)

        cls_token_final = x[:, 0] # just the CLS token
        x = self.head(cls_token_final)

        return x


        








