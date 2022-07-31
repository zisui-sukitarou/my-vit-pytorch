"""
    - 解説記事 :
        - https://qiita.com/zisui-sukitarou/items/d990a9630ff2c7f4abf2

    - 参考 :
        - [1] https://arxiv.org/abs/2010.11929
        - [2] https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
"""

import torch
import torch.nn as nn

from einops import repeat
from einops.layers.torch import Rearrange


class Patching(nn.Module):
    def __init__(self, patch_size):
        """ [input]
            - patch_size (int) : パッチの縦の長さ（=横の長さ）
        """
        super().__init__()
        self.net = Rearrange("b c (h ph) (w pw) -> b (h w) (ph pw c)", ph = patch_size, pw = patch_size)
    
    def forward(self, x):
        """ [input]
            - x (torch.Tensor) : 画像データ
                - x.shape = torch.Size([batch_size, channels, image_height, image_width])
        """
        x = self.net(x)
        return x


class LinearProjection(nn.Module):
    def __init__(self, patch_dim, dim):
        """ [input]
            - patch_dim (int) : 一枚あたりのパッチのベクトルの長さ（= channels * (patch_size ** 2)）
            - dim (int) : パッチのベクトルが変換されたベクトルの長さ 
        """
        super().__init__()
        self.net = nn.Linear(patch_dim, dim)

    def forward(self, x):
        """ [input]
            - x (torch.Tensor) 
                - x.shape = torch.Size([batch_size, n_patches, patch_dim])
        """
        x = self.net(x)
        return x


class Embedding(nn.Module):
    def __init__(self, dim, n_patches):
        """ [input]
            - dim (int) : パッチのベクトルが変換されたベクトルの長さ
            - n_patches (int) : パッチの枚数
        """
        super().__init__()
        # [class] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # position embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, n_patches + 1, dim))
    
    def forward(self, x):
        """[input]
            - x (torch.Tensor)
                - x.shape = torch.Size([batch_size, n_patches, dim])
        """
        # バッチサイズを抽出
        batch_size, _, __ = x.shape

        # [class] トークン付加
        # x.shape : [batch_size, n_patches, patch_dim] -> [batch_size, n_patches + 1, patch_dim]
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b = batch_size)
        x = torch.concat([cls_tokens, x], dim = 1)

        # 位置エンコーディング
        x += self.pos_embedding

        return x


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        """ [input]
            - dim (int) : パッチのベクトルが変換されたベクトルの長さ
            - hidden_dim (int) : 隠れ層のノード数
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        """[input]
            - x (torch.Tensor)
                - x.shape = torch.Size([batch_size, n_patches + 1, dim])
        """
        x = self.net(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads):
        """ [input]
            - dim (int) : パッチのベクトルが変換されたベクトルの長さ
            - n_heads (int) : heads の数
        """
        super().__init__()
        self.n_heads = n_heads
        self.dim_heads = dim // n_heads

        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)

        self.split_into_heads = Rearrange("b n (h d) -> b h n d", h = self.n_heads)

        self.softmax = nn.Softmax(dim = -1)

        self.concat = Rearrange("b h n d -> b n (h d)", h = self.n_heads)

    def forward(self, x):
        """[input]
            - x (torch.Tensor)
                - x.shape = torch.Size([batch_size, n_patches + 1, dim])
        """
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        q = self.split_into_heads(q)
        k = self.split_into_heads(k)
        v = self.split_into_heads(v)

        # Logit[i] = Q[i] * tK[i] / sqrt(D) (i = 1, ... , n_heads)
        # AttentionWeight[i] = Softmax(Logit[i]) (i = 1, ... , n_heads)
        logit = torch.matmul(q, k.transpose(-1, -2)) * (self.dim_heads ** -0.5)
        attention_weight = self.softmax(logit)

        # Head[i] = AttentionWeight[i] * V[i] (i = 1, ... , n_heads)
        # Output = concat[Head[1], ... , Head[n_heads]]
        output = torch.matmul(attention_weight, v)
        output = self.concat(output)
        return output


class TransformerEncoder(nn.Module):
    def __init__(self, dim, n_heads, mlp_dim, depth):
        """ [input]
            - dim (int) : 各パッチのベクトルが変換されたベクトルの長さ（参考[1] (1)式 D）
            - depth (int) : Transformer Encoder の層の深さ（参考[1] (2)式 L）
            - n_heads (int) : Multi-Head Attention の head の数
            - mlp_dim (int) : MLP の隠れ層のノード数
        """
        super().__init__()

        # Layers
        self.norm = nn.LayerNorm(dim)
        self.multi_head_attention = MultiHeadAttention(dim = dim, n_heads = n_heads)
        self.mlp = MLP(dim = dim, hidden_dim = mlp_dim)
        self.depth = depth

    def forward(self, x):
        """[input]
            - x (torch.Tensor)
                - x.shape = torch.Size([batch_size, n_patches + 1, dim])
        """
        for _ in range(self.depth):
            x = self.multi_head_attention(self.norm(x)) + x
            x = self.mlp(self.norm(x)) + x

        return x


class MLPHead(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_dim)
        )
    
    def forward(self, x):
        x = self.net(x)
        return x


class ViT(nn.Module):
    def __init__(self, image_size, patch_size, n_classes, dim, depth, n_heads, channels = 3, mlp_dim = 256):
        """ [input]
            - image_size (int) : 画像の縦の長さ（= 横の長さ）
            - patch_size (int) : パッチの縦の長さ（= 横の長さ）
            - n_classes (int) : 分類するクラスの数
            - dim (int) : 各パッチのベクトルが変換されたベクトルの長さ（参考[1] (1)式 D）
            - depth (int) : Transformer Encoder の層の深さ（参考[1] (2)式 L）
            - n_heads (int) : Multi-Head Attention の head の数
            - chahnnels (int) : 入力のチャネル数（RGBの画像なら3）
            - mlp_dim (int) : MLP の隠れ層のノード数
        """

        super().__init__()
        
        # Params
        n_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size * patch_size
        self.depth = depth

        # Layers
        self.patching = Patching(patch_size = patch_size)
        self.linear_projection_of_flattened_patches = LinearProjection(patch_dim = patch_dim, dim = dim)
        self.embedding = Embedding(dim = dim, n_patches = n_patches)
        self.transformer_encoder = TransformerEncoder(dim = dim, n_heads = n_heads, mlp_dim = mlp_dim, depth = depth)
        self.mlp_head = MLPHead(dim = dim, out_dim = n_classes)


    def forward(self, img):
        """ [input]
            - img (torch.Tensor) : 画像データ
                - img.shape = torch.Size([batch_size, channels, image_height, image_width])
        """

        x = img

        # 1. パッチに分割
        # x.shape : [batch_size, channels, image_height, image_width] -> [batch_size, n_patches, channels * (patch_size ** 2)]
        x = self.patching(x)

        # 2. 各パッチをベクトルに変換
        # x.shape : [batch_size, n_patches, channels * (patch_size ** 2)] -> [batch_size, n_patches, dim]
        x = self.linear_projection_of_flattened_patches(x)

        # 3. [class] トークン付加 + 位置エンコーディング 
        # x.shape : [batch_size, n_patches, dim] -> [batch_size, n_patches + 1, dim]
        x = self.embedding(x)

        # 4. Transformer Encoder
        # x.shape : No Change
        x = self.transformer_encoder(x)

        # 5. 出力の0番目のベクトルを MLP Head で処理
        # x.shape : [batch_size, n_patches + 1, dim] -> [batch_size, dim] -> [batch_size, n_classes]
        x = x[:, 0]
        x = self.mlp_head(x)

        return x