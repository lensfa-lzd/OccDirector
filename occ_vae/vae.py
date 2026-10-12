"""
Modified from https://github.com/gusongen/DOME/tree/da390aeeab1990fa06e037ad4d486729ae4fe712/model/VAE
"""
from typing import Tuple, Dict

import torch
from einops import repeat, rearrange
from torch import nn

from occ_vae.module import ResnetBlock, AttnBlock, Downsample, Normalize, nonlinearity, Upsample


class DiagonalGaussian(nn.Module):
    def __init__(self, sample: bool = True, chunk_dim: int = 1):
        super().__init__()
        self.sample = sample
        self.chunk_dim = chunk_dim

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # z [bs, channel, dim1, dim2, ...]
        mean, logvar = torch.chunk(z, 2, dim=self.chunk_dim)
        if self.sample:
            std = torch.exp(0.5 * logvar)
            return mean + std * torch.randn_like(mean), mean, logvar
        else:
            return mean, mean, logvar


class OccWarper(nn.Module):
    def __init__(
            self,
            occ_depth,  # Occ 场景的高度
            num_classes,
            expansion,  # 类别 Embedding 维度
    ):
        super().__init__()
        self.expansion = expansion
        self.class_embeds = nn.Embedding(num_classes, expansion)
        self.occ_emb_channels = occ_depth * expansion

    def class_emb(self, x: torch.Tensor) -> torch.Tensor:
        # x: bs, H, W, D [1, 512, 512, 16]
        if len(x.shape) == 4:
            # x [bs, height, weight, depth]
            # 填充 frame
            x = repeat(x, "bs h w d -> bs f h w d", f=1).contiguous()

        # assert len(x.shape) == 5, "The input tensor should have 5 dimensions."
        x = self.class_embeds(x.int())
        x = rearrange(x, 'bs f h w d c -> bs (d c) f h w').contiguous()
        return x

    def reverse_class_emb(self, hidden_logits):
        f, h, w = hidden_logits.shape[-3:]
        hidden_logits = rearrange(
            hidden_logits,
            'bs (d expansion) f h w -> (bs f h w) d expansion',
            expansion=self.expansion
        ).contiguous()

        # template 的形状: (1, expansion, num_cls)
        template = self.class_embeds.weight.T.unsqueeze(0)

        # similarity 的形状: (bs * f * h * w, D, num_cls)。
        # 含义: 对于每个空间位置 (bs*f*h*w)，对于序列中的每个位置 (D)，我们都计算出了一个长度为 num_cls 的向量。
        # 这个向量里的每个值，代表该位置的解码特征与词典中对应词元的相似度分数（或 logits）。
        similarity = torch.matmul(hidden_logits, template)  # -1, D, cls
        logits = rearrange(
            similarity,
            '(bs f h w) d num_cls -> bs num_cls f h w d',
            f=f,
            h=h,
            w=w
        ).contiguous()
        return logits


class Encoder2D(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,

            att_level: Tuple[int, ...],  # 添加注意力的层
            block_out_channels: Tuple[int, ...],  # 每个 block 的输出维度
            num_res_blocks: int,

            dropout=0.0,
    ):
        super().__init__()
        # downsampling
        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.down = nn.ModuleList()
        block_out = block_out_channels[0]
        for i_level in range(len(block_out_channels)):
            block = nn.Sequential()

            block_in = block_out
            block_out = block_out_channels[i_level]
            for i_block in range(num_res_blocks):
                block.append(ResnetBlock(
                    in_channels=block_in,
                    out_channels=block_out,
                    dropout=dropout,
                    conv_module=nn.Conv2d,
                ))
                if i_level in att_level:
                    block.append(AttnBlock(
                        channels=block_out,
                        conv_module=nn.Conv2d,
                    ))
                else:
                    block.append(nn.Identity())

                block_in = block_out

            down = nn.Module()
            down.block = block
            if i_level != len(block_out_channels) - 1:
                down.downsample = Downsample(channels=block_out)
            else:
                down.downsample = nn.Identity()

            self.down.append(down)

        # middle
        mid_channel = block_out_channels[-1]
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=mid_channel,
            out_channels=mid_channel,
            dropout=dropout,
            conv_module=nn.Conv2d,
        )
        self.mid.attn_1 = AttnBlock(
            channels=mid_channel,
            conv_module=nn.Conv2d
        )
        self.mid.block_2 = ResnetBlock(
            in_channels=mid_channel,
            out_channels=mid_channel,
            dropout=dropout,
            conv_module=nn.Conv2d,
        )

        # end
        self.norm_out = Normalize(channels=mid_channel)
        self.conv_out = nn.Conv2d(
            in_channels=mid_channel,
            out_channels=out_channels * 2,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x):
        # x: bs, F, H, W, D
        x_shape = x.shape
        x = rearrange(x, "b c f h w -> (b f) c h w")

        h = self.conv_in(x)
        for down_block in self.down:
            h = down_block.block(h)
            h = down_block.downsample(h)

        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        b, _, f, _, _ = x_shape
        h = rearrange(h, "(b f) c h w -> b c f h w", b=b, f=f)
        return h


class Decoder3D(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,

            att_level: Tuple[int, ...],  # 添加注意力的层
            block_out_channels: Tuple[int, ...],  # 每个 block 的输出维度
            num_res_blocks: int,

            dropout=0.0,

    ):
        super().__init__()
        # z to block_in
        self.conv_in = nn.Conv3d(
            in_channels=in_channels,
            out_channels=block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1
        )

        # middle
        mid_channel = block_out_channels[0]
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=mid_channel,
            out_channels=mid_channel,
            dropout=dropout,
            conv_module=nn.Conv3d,
        )
        self.mid.attn_1 = AttnBlock(
            channels=mid_channel,
            conv_module=nn.Conv3d,
        )
        self.mid.block_2 = ResnetBlock(
            in_channels=mid_channel,
            out_channels=mid_channel,
            dropout=dropout,
            conv_module=nn.Conv3d,
        )

        # upsampling
        self.up = nn.ModuleList()
        block_out = block_out_channels[0]
        for i_level in range(len(block_out_channels)):
            block = nn.Sequential()

            block_in = block_out
            block_out = block_out_channels[i_level]
            for i_block in range(num_res_blocks):
                block.append(ResnetBlock(
                    in_channels=block_in,
                    out_channels=block_out,
                    dropout=dropout,
                    conv_module=nn.Conv3d
                ))
                if i_level in att_level:
                    block.append(AttnBlock(
                        channels=block_out,
                        conv_module=nn.Conv3d,
                    ))
                else:
                    block.append(nn.Identity())

                block_in = block_out

            up = nn.Module()
            up.block = block

            # 最外层 （最靠近输出）没有 upsample 操作
            if i_level != len(block_out_channels) - 1:
                up.upsample = Upsample(block_out)
            else:
                up.upsample = nn.Identity()

            self.up.append(up)

        # end
        end_channel = block_out_channels[-1]
        self.norm_out = Normalize(end_channel)
        self.conv_out = nn.Conv3d(
            end_channel,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, z):
        # x: bs, C, F, H, W

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for up_block in self.up:
            h = up_block.block(h)
            h = up_block.upsample(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class OccSeqCausalVAE(nn.Module):
    def __init__(
            self,
            num_classes: int = 11,
            expansion: int = 8,  # 类别 Embedding 维度
            occ_depth: int = 16,  # Occ 场景的高度
            occ_latent_channels: int = 64,

            use_scale: bool = False,
            scale_mean: float = None,
            scale_std: float = None,
    ):
        super().__init__()
        self.reg = DiagonalGaussian()
        self.occ_warper = OccWarper(
            occ_depth=occ_depth,
            num_classes=num_classes,
            expansion=expansion,
        )
        in_channels = out_channels = self.occ_warper.occ_emb_channels  # Occ 嵌入后的维度

        # 单帧 latent 的空间形状（输入 224 时为 28×28），OccGen 的 MMDiT 由此构造 patch 布局
        self.latent_occ_frame_shape = (28, 28, occ_latent_channels)

        self.encoder = Encoder2D(
            in_channels=in_channels,
            out_channels=occ_latent_channels,
            att_level=(2,),
            block_out_channels=(64, 128, 256, 512),
            num_res_blocks=2,
        )
        self.decoder = Decoder3D(
            in_channels=occ_latent_channels,
            out_channels=out_channels,
            att_level=(1,),
            block_out_channels=(512, 256, 128, 64),
            num_res_blocks=2,
        )

        self.use_scale = use_scale
        if self.use_scale:
            self.scale_mean = scale_mean
            self.scale_std = scale_std

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode a batch of 3D/4D Occ into latents.
        """
        x = self.occ_warper.class_emb(x)

        h = self.encoder(x)
        z, mean, logvar = self.reg(h)

        if self.use_scale:
            z = (z - self.scale_mean) / self.scale_std

        return z, mean, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """将 VAE latent 解码为语义类别 logits。

        Args:
            z (torch.Tensor): latent 张量，形状为 `[B, C_lat, F, H_lat, W_lat]`。

        Returns:
            torch.Tensor: 语义类别 logits，形状为 `[B, num_classes, F, H, W, D]`。
            其中 `D` 为垂直方向 depth bins。
        """
        if self.use_scale:
            z = z * self.scale_std + self.scale_mean

        hidden_logits = self.decoder(z)
        logits = self.occ_warper.reverse_class_emb(hidden_logits)
        return logits

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z, mean, logvar = self.encode(x)
        logits = self.decode(z)
        return {
            'logits': logits,
            'z_mu': mean,
            'z_logvar': logvar
        }

    def freeze_grad(self) -> None:
        """冻结 VAE 的所有参数（不参与训练）。"""
        for param in self.parameters():
            param.requires_grad = False


def get_vae(use_scale: bool = False, ckpt_path: str = '') -> OccSeqCausalVAE:
    """构建并（可选）从 checkpoint 加载 VAE。

    Args:
        use_scale (bool): 是否对 latent 做线性缩放（使用 checkpoint 中保存的 mean/std）。
        ckpt_path (str): checkpoint 路径。为空字符串表示不加载。

    Returns:
        OccSeqCausalVAE: VAE 模型。
    """

    if ckpt_path != '':
        # 需要是已经整理好的权重
        print(f'Loading vae checkpoint {ckpt_path}')
        ckpt = torch.load(ckpt_path, weights_only=True, map_location='cpu')

        scale_factor = ckpt['ema_state_dict_scale_factor']
        vae = OccSeqCausalVAE(
            use_scale=use_scale,
            scale_mean=scale_factor['mean'],
            scale_std=scale_factor['std'],
        )
        vae.load_state_dict(ckpt['ema_state_dict'])
    else:
        vae = OccSeqCausalVAE()
    return vae


if __name__ == '__main__':
    vae = get_vae()

    train_params = 0
    total_params = 0
    for p in vae.parameters():
        total_params += p.numel()
        if p.requires_grad:
            train_params += p.numel()

    print(f"Train params: {train_params:,}; Total params: {total_params:,}")

    data = torch.ones((4, 8, 224, 224, 16))
    re_data = vae(data)
    print()
