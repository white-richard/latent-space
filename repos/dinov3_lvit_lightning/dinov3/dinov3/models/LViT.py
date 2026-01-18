# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
import math
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
import torch.nn.init
from torch import Tensor, nn

from hypercore import nn as hnn
from hypercore.manifolds import Lorentz

from dinov3.layers import RopePositionEmbedding
from dinov3.utils import named_apply

logger = logging.getLogger("dinov3")

dtype_dict = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _make_2tuple(x: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(x, tuple):
        assert len(x) == 2
        return x
    assert isinstance(x, int)
    return (x, x)


def _get_c(manifold: Lorentz, like: Tensor) -> Tensor:
    c = manifold.c
    if torch.is_tensor(c):
        return c.to(device=like.device, dtype=like.dtype)
    return like.new_tensor(c)


def _space_to_lorentz(manifold: Lorentz, x_space: Tensor) -> Tensor:
    c = _get_c(manifold, x_space)
    x_time = ((x_space**2).sum(dim=-1, keepdim=True) + c).clamp_min(1e-6).sqrt()
    return torch.cat([x_time, x_space], dim=-1)


def _rope_to_complex(rope: Tuple[Tensor, Tensor]) -> Tensor:
    sin, cos = rope
    if sin.numel() == 0:
        return sin.new_zeros((*sin.shape[:-1], 0), dtype=torch.complex64)
    half = sin.shape[-1] // 2
    sin = sin[..., :half].to(dtype=torch.float32)
    cos = cos[..., :half].to(dtype=torch.float32)
    return torch.complex(cos, sin)


def _pad_rope_prefix(freqs: Tensor, prefix_tokens: int) -> Tensor:
    if prefix_tokens <= 0:
        return freqs
    ones = torch.ones(
        (prefix_tokens, freqs.shape[-1]), device=freqs.device, dtype=freqs.dtype
    )
    return torch.cat([ones, freqs], dim=0)


class SafeLorentzMLR(nn.Module):
    def __init__(
        self, manifold: Lorentz, num_features: int, num_classes: int, eps: float = 1e-4
    ):
        super().__init__()
        self.manifold = manifold
        self.a = nn.Parameter(torch.zeros(num_classes))
        self.z = nn.Parameter(
            F.pad(torch.zeros(num_classes, num_features - 2), pad=(1, 0), value=1)
        )
        self.c = manifold.c
        self.eps = eps
        self.init_weights()

    def init_weights(self) -> None:
        stdv = 1.0 / math.sqrt(self.z.size(1))
        nn.init.uniform_(self.z, -stdv, stdv)
        nn.init.uniform_(self.a, -stdv, stdv)

    def forward(self, x: Tensor) -> Tensor:
        sqrt_mk = 1 / self.c.sqrt()
        norm_z = torch.norm(self.z, dim=-1).clamp_min(self.eps)
        w_t = torch.sinh(sqrt_mk * self.a) * norm_z
        w_s = torch.cosh(sqrt_mk * self.a.view(-1, 1)) * self.z
        beta_sq = (-(w_t**2) + torch.norm(w_s, dim=-1) ** 2).clamp_min(self.eps)
        beta = torch.sqrt(beta_sq)
        alpha = -w_t * x.narrow(-1, 0, 1) + (
            torch.cosh(sqrt_mk * self.a)
            * torch.inner(x.narrow(-1, 1, x.shape[-1] - 1), self.z)
        )
        alpha_over_beta = alpha / beta.clamp_min(self.eps)

        # Clamp alpha_over_beta to avoid asinh overflow/instability
        alpha_over_beta = alpha_over_beta.clamp(-1e4, 1e4)

        d = self.c.sqrt() * torch.abs(torch.asinh(sqrt_mk * alpha_over_beta))
        logits = torch.sign(alpha_over_beta) * beta * d
        return torch.nan_to_num(logits)


def init_weights_lvit(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    if isinstance(module, hnn.LorentzLinear):
        module.reset_parameters()
    if isinstance(module, hnn.LResNet):
        # Reinitialize residual scaling params overwritten by NaN init.
        if hasattr(module, "w_y") and torch.is_tensor(module.w_y):
            module.w_y.data.fill_(1.0)
        if hasattr(module, "scale") and torch.is_tensor(module.scale):
            module.scale.data.zero_()
    if isinstance(module, hnn.LorentzMultiheadAttention):
        if hasattr(module, "scale") and torch.is_tensor(module.scale):
            module.scale.data.fill_(math.sqrt(module.num_heads * module.out_channels))
        if hasattr(module, "bias") and torch.is_tensor(module.bias):
            module.bias.data.zero_()
        if hasattr(module, "norm_scale") and torch.is_tensor(module.norm_scale):
            module.norm_scale.data.fill_(1.0)
    if isinstance(module, hnn.LorentzConv2d):
        module.reset_parameters()
    if isinstance(module, hnn.LorentzLayerNorm):
        module.reset_parameters()
    if isinstance(module, hnn.LorentzRMSNorm):
        if hasattr(module, "weight"):
            module.weight.data.fill_(1.0)


class LorentzPatchEmbed(nn.Module):
    """
    Hyperbolic patch embedding: (B, C, H, W) -> (B, H, W, D) or (B, N, D)
    """

    def __init__(
        self,
        *,
        manifold: Lorentz,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[nn.Module] = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_hw = _make_2tuple(img_size)
        patch_hw = _make_2tuple(patch_size)
        patch_grid_size = (image_hw[0] // patch_hw[0], image_hw[1] // patch_hw[1])

        self.manifold = manifold
        self.img_size = image_hw
        self.patch_size = patch_hw
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten_embedding = flatten_embedding

        if embed_dim < 2:
            raise ValueError("embed_dim must be >= 2 for Lorentz coordinates.")

        self.proj = hnn.LorentzConv2d(
            manifold_in=manifold,
            in_channels=in_chans + 1,
            out_channels=embed_dim - 1,
            kernel_size=patch_hw,
            stride=patch_hw,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        # Hyperbolic lift: compute time coordinate to place pixels on the Lorentz manifold.
        x = _space_to_lorentz(self.manifold, x)
        x = self.proj(x)
        x = self.norm(x)
        if self.flatten_embedding:
            x = x.flatten(1, 2)
        return x

    def reset_parameters(self):
        self.proj.reset_parameters()


class LorentzMlp(nn.Module):
    def __init__(
        self,
        *,
        manifold: Lorentz,
        in_features: int,
        hidden_features: int,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = hnn.LorentzLinear(
            manifold, in_features, hidden_features - 1, bias=bias
        )
        self.act = hnn.LorentzActivation(manifold, activation=act_layer())
        self.fc2 = hnn.LorentzLinear(
            manifold, hidden_features, out_features - 1, bias=bias
        )
        self.drop = hnn.LorentzDropout(manifold, drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LorentzSwiGLUFFN(nn.Module):
    def __init__(
        self,
        *,
        manifold: Lorentz,
        in_features: int,
        hidden_features: int,
        out_features: Optional[int] = None,
        drop: float = 0.0,
        bias: bool = True,
        align_to: Optional[int] = None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        if align_to:
            hidden_features = int(math.ceil(hidden_features / align_to) * align_to)
        self.manifold = manifold
        self.w1 = hnn.LorentzLinear(
            manifold, in_features, hidden_features - 1, bias=bias
        )
        self.w3 = hnn.LorentzLinear(
            manifold, in_features, hidden_features - 1, bias=bias
        )
        self.w2 = hnn.LorentzLinear(
            manifold, hidden_features, out_features - 1, bias=bias
        )
        self.drop = hnn.LorentzDropout(manifold, drop)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.w1(x, return_space=True)
        x3 = self.w3(x, return_space=True)
        x_space = F.silu(x1) * x3
        # Hyperbolic rebuild: recompute time coordinate after gating.
        x = _space_to_lorentz(self.manifold, x_space)
        x = self.w2(x)
        x = self.drop(x)
        return x


class LorentzMultiheadAttention(hnn.LorentzMultiheadAttention):
    def apply_rotary_embeddings(
        self, x: Tensor, freqs_complex: Tensor, device: torch.device
    ) -> Tensor:
        if freqs_complex is None or freqs_complex.numel() == 0:
            return x
        rot_dim = freqs_complex.shape[-1] * 2
        max_rot_dim = x.shape[-1] - (x.shape[-1] % 2)
        if max_rot_dim <= 0:
            return x
        rot_dim = min(rot_dim, max_rot_dim)
        freqs_complex = freqs_complex[..., : rot_dim // 2]
        if rot_dim <= 0:
            return x
        x_rot = x[..., :rot_dim]
        x_pass = x[..., rot_dim:]
        # Apply axial RoPE to spatial coordinates only; time is rebuilt downstream.
        x_rot = x_rot.float().contiguous()
        x_complex = torch.view_as_complex(x_rot.reshape(*x_rot.shape[:-1], -1, 2))
        freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
        x_rotated = x_complex * freqs_complex
        x_out = torch.view_as_real(x_rotated).reshape(*x_rot.shape)
        x_out = x_out.type_as(x).to(device)
        if x_pass.numel() == 0:
            return x_out
        return torch.cat([x_out, x_pass], dim=-1)


class LorentzSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        manifold: Lorentz,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: Optional[nn.Module] = None,
        ffn_layer: Optional[nn.Module] = None,
        residual_scale: Optional[float] = 27.5,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by num_heads for Lorentz attention."
            )

        self.dim = dim
        self.num_heads = num_heads
        self.sample_drop_ratio = drop_path

        head_dim = dim // num_heads
        self.norm1 = norm_layer(dim)
        self.attn = LorentzMultiheadAttention(
            manifold,
            head_dim,
            head_dim,
            num_heads,
            attention_type="full",
            trans_heads_concat=True,
        )
        self.residual_1 = hnn.LResNet(
            manifold, use_scale=residual_scale is not None, scale=residual_scale
        )

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * ffn_ratio)
        self.mlp = ffn_layer(
            manifold=manifold,
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.residual_2 = hnn.LResNet(
            manifold, use_scale=residual_scale is not None, scale=residual_scale
        )

        if init_values is not None:
            self.residual_1.w_y.data.fill_(init_values)
            self.residual_2.w_y.data.fill_(init_values)

    def _apply_residual(
        self, x: Tensor, residual: Tensor, block: hnn.LResNet
    ) -> Tensor:
        if not self.training or self.sample_drop_ratio <= 0.0:
            return block(x, residual)
        keep_prob = 1.0 - self.sample_drop_ratio
        if keep_prob <= 0.0:
            return block(
                x,
                residual,
                weight=torch.zeros((x.shape[0], 1, 1), device=x.device, dtype=x.dtype),
            )
        drop_mask = (torch.rand((x.shape[0], 1, 1), device=x.device) < keep_prob).to(
            x.dtype
        )
        weight = block.w_y.to(x.dtype) * drop_mask / keep_prob
        return block(x, residual, weight=weight)

    def _forward(self, x: Tensor, rope=None) -> Tensor:
        # Lorentz pre-norm + hyperbolic self-attention.
        x_norm = self.norm1(x)
        x_attn = self.attn(x_norm, x_norm, rot_pos=rope)
        x = self._apply_residual(x, x_attn, self.residual_1)
        x_mlp = self.mlp(self.norm2(x))
        x = self._apply_residual(x, x_mlp, self.residual_2)
        return x

    def forward(self, x_or_x_list, rope_or_rope_list=None):
        if isinstance(x_or_x_list, Tensor):
            return self._forward(x_or_x_list, rope=rope_or_rope_list)
        if rope_or_rope_list is None:
            rope_or_rope_list = [None for _ in x_or_x_list]
        return [
            self._forward(x, rope=rope)
            for x, rope in zip(x_or_x_list, rope_or_rope_list)
        ]


def _build_norm_layer(norm_layer: str, manifold: Lorentz):
    if norm_layer == "layernorm":
        return lambda dim: hnn.LorentzLayerNorm(manifold, dim - 1, eps=1e-6)
    if norm_layer == "layernormbf16":
        return lambda dim: hnn.LorentzLayerNorm(manifold, dim - 1, eps=1e-5)
    if norm_layer == "rmsnorm":
        return lambda dim: hnn.LorentzRMSNorm(manifold, dim - 1, eps=1e-4)
    raise ValueError(f"Unknown norm_layer: {norm_layer}")


ffn_layer_dict = {
    "mlp": LorentzMlp,
    "swiglu": LorentzSwiGLUFFN,
    "swiglu32": partial(LorentzSwiGLUFFN, align_to=32),
    "swiglu64": partial(LorentzSwiGLUFFN, align_to=64),
    "swiglu128": partial(LorentzSwiGLUFFN, align_to=128),
}


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_min_period: float | None = None,
        pos_embed_rope_max_period: float | None = None,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        pos_embed_rope_shift_coords: float | None = None,
        pos_embed_rope_jitter_coords: float | None = None,
        pos_embed_rope_rescale_coords: float | None = None,
        pos_embed_rope_dtype: str = "bf16",
        num_classes: int = 0,
        embed_dim: int | None = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: float | None = None,
        norm_layer: str = "layernorm",
        ffn_layer: str = "mlp",
        ffn_bias: bool = True,
        proj_bias: bool = True,
        n_storage_tokens: int = 0,
        mask_k_bias: bool = False,
        untie_cls_and_patch_norms: bool = False,
        untie_global_and_local_cls_norm: bool = False,
        device: Any | None = None,
        **ignored_kwargs,
    ):
        super().__init__()

        image_size = ignored_kwargs.pop("image_size", None)
        if image_size is not None:
            img_size = image_size
        num_layers = ignored_kwargs.pop("num_layers", None)
        if num_layers is not None:
            depth = num_layers
        in_channel = ignored_kwargs.pop("in_channel", None)
        if in_channel is not None:
            in_chans = in_channel
        
        manifold_in = ignored_kwargs.pop("manifold_in", None)
        manifold_hidden = ignored_kwargs.pop("manifold_hidden", None)
        manifold_out = ignored_kwargs.pop("manifold_out", None)
        manifold = (
            ignored_kwargs.pop("manifold", None)
            or manifold_in
            or manifold_hidden
            or manifold_out
            or Lorentz()
        )

        mlp_hidden_expansion = ignored_kwargs.pop("mlp_hidden_expansion", None)
        if mlp_hidden_expansion is not None:
            ffn_ratio = mlp_hidden_expansion

        self.num_classes = num_classes
        dropout = ignored_kwargs.pop("dropout", 0.0)
        residual_scale = ignored_kwargs.pop("residual_scale", 27.5)
        pos_embed_type = ignored_kwargs.pop("pos_embed_type", "rope")

        hidden_channel = ignored_kwargs.pop("hidden_channel", None)
        if embed_dim is None and hidden_channel is not None:
            embed_dim = hidden_channel * num_heads
        if embed_dim is None:
            embed_dim = 768

        # Not sure if this should be calculated or just 64
        if hidden_channel is None:
            if embed_dim % num_heads != 0:
                raise ValueError(
                    f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
                )
            hidden_channel = embed_dim // num_heads
        if manifold_out is None:
            manifold_out = manifold

        if len(ignored_kwargs) > 0:
            logger.warning(f"Ignored kwargs: {ignored_kwargs}")
        del ignored_kwargs

        norm_layer_cls = _build_norm_layer(norm_layer, manifold)
        if ffn_layer not in ffn_layer_dict:
            raise ValueError(f"Unknown ffn_layer: {ffn_layer}")
        ffn_layer_cls = ffn_layer_dict[ffn_layer]

        self.num_features = self.embed_dim = embed_dim
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.n_storage_tokens = n_storage_tokens
        self.manifold = manifold
        self.manifold_out = manifold_out
        self.hidden_channel = hidden_channel
        if hasattr(self.manifold, "c"):
            c_val = self.manifold.c
            self._manifold_k_init = float(c_val.detach().cpu() if torch.is_tensor(c_val) else c_val)
        else:
            self._manifold_k_init = 1.0
        if hasattr(self.manifold_out, "c"):
            c_val = self.manifold_out.c
            self._manifold_out_k_init = float(c_val.detach().cpu() if torch.is_tensor(c_val) else c_val)
        else:
            self._manifold_out_k_init = 1.0

        self.patch_embed = LorentzPatchEmbed(
            manifold=manifold,
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )

        self.cls_token_space = nn.Parameter(
            torch.empty(1, 1, embed_dim - 1, device=device)
        )
        if self.n_storage_tokens > 0:
            self.storage_tokens_space = nn.Parameter(
                torch.empty(1, n_storage_tokens, embed_dim - 1, device=device)
            )
        else:
            self.storage_tokens_space = None

        self.pos_embed_type = pos_embed_type
        self.rope_dim = 0
        if pos_embed_type == "rope":
            head_dim = embed_dim // num_heads
            space_dim = head_dim - 1
            rope_dim = space_dim - (space_dim % 4)
            self.rope_dim = max(rope_dim, 0)
            if self.rope_dim <= 0:
                logger.warning(
                    "RoPE disabled: spatial head dimension too small for axial RoPE."
                )
                self.rope_embed = None
            else:
                if self.rope_dim < space_dim:
                    logger.warning(
                        "RoPE rotates %d/%d spatial dims per head (remaining dims left unrotated).",
                        self.rope_dim,
                        space_dim,
                    )
                self.rope_embed = RopePositionEmbedding(
                    embed_dim=self.rope_dim * num_heads,
                    num_heads=num_heads,
                    base=pos_embed_rope_base,
                    min_period=pos_embed_rope_min_period,
                    max_period=pos_embed_rope_max_period,
                    normalize_coords=pos_embed_rope_normalize_coords,
                    shift_coords=pos_embed_rope_shift_coords,
                    jitter_coords=pos_embed_rope_jitter_coords,
                    rescale_coords=pos_embed_rope_rescale_coords,
                    dtype=dtype_dict[pos_embed_rope_dtype],
                    device=device,
                )
            self.use_pos_embed = False
            self.pos_embed_space = None
            self.pos_embed_add = None
            self.pos_embed_hw = None
        elif pos_embed_type == "learned":
            self.use_pos_embed = True
            grid_h, grid_w = self.patch_embed.patches_resolution
            self.pos_embed_space = nn.Parameter(
                torch.empty(1, grid_h, grid_w, embed_dim - 1, device=device)
            )
            self.pos_embed_add = hnn.LResNet(manifold, use_scale=True, scale=1.0)
            self.pos_embed_hw = (grid_h, grid_w)
            self.rope_embed = None
        elif pos_embed_type in ("none", None):
            self.use_pos_embed = False
            self.pos_embed_space = None
            self.pos_embed_add = None
            self.pos_embed_hw = None
            self.rope_embed = None
        else:
            raise ValueError(f"Unknown pos_embed_type: {pos_embed_type}")

        print(f"WARNING: Must call init_weights() after creating!!")

        blocks_list = [
            LorentzSelfAttentionBlock(
                manifold=manifold,
                dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop=dropout,
                init_values=layerscale_init,
                drop_path=drop_path_rate,
                act_layer=nn.GELU,
                norm_layer=norm_layer_cls,
                ffn_layer=ffn_layer_cls,
                residual_scale=residual_scale,
            )
            for _ in range(depth)
        ]

        self.chunked_blocks = False
        self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer_cls(embed_dim)

        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        if untie_cls_and_patch_norms:
            self.cls_norm = norm_layer_cls(embed_dim)
        else:
            self.cls_norm = None

        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm
        if untie_global_and_local_cls_norm:
            self.local_cls_norm = norm_layer_cls(embed_dim)
        else:
            self.local_cls_norm = None

        if self.num_classes > 0:
            self.head = SafeLorentzMLR(
                manifold_out,
                self.num_heads * hidden_channel,
                self.num_classes,
            )
        else:
            self.head = nn.Identity()
        self.mask_token_space = nn.Parameter(
            torch.empty(1, embed_dim - 1, device=device)
        )

    def init_weights(self):
        if hasattr(self.manifold, "k") and torch.is_tensor(self.manifold.k):
            self.manifold.k.data.copy_(self.manifold.k.new_tensor(self._manifold_k_init))
        if hasattr(self.manifold_out, "k") and torch.is_tensor(self.manifold_out.k):
            self.manifold_out.k.data.copy_(self.manifold_out.k.new_tensor(self._manifold_out_k_init))
        if self.rope_embed is not None:
            self.rope_embed._init_weights()
        nn.init.normal_(self.cls_token_space, std=0.02)
        if self.storage_tokens_space is not None:
            nn.init.normal_(self.storage_tokens_space, std=0.02)
        if self.pos_embed_space is not None:
            nn.init.normal_(self.pos_embed_space, std=0.02)
        nn.init.zeros_(self.mask_token_space)
        named_apply(init_weights_lvit, self)

    def _build_token(self, token_space: Tensor) -> Tensor:
        # Hyperbolic rebuild: ensure learnable spatial tokens satisfy the Lorentz constraint.
        return _space_to_lorentz(self.manifold, token_space)

    def _build_rope(self, H: int, W: int, device: torch.device) -> Optional[Tensor]:
        if self.rope_embed is None:
            return None
        rope = self.rope_embed(H=H, W=W)
        freqs = _rope_to_complex(rope).to(device)
        # Prefix tokens (CLS/storage) get identity rotation for RoPE.
        return _pad_rope_prefix(freqs, self.n_storage_tokens + 1)

    def _get_pos_embed(self, H: int, W: int) -> Optional[Tensor]:
        if self.pos_embed_space is None:
            return None
        pos_space = self.pos_embed_space
        if (H, W) != self.pos_embed_hw:
            pos_space = pos_space.permute(0, 3, 1, 2)
            pos_space = F.interpolate(
                pos_space, size=(H, W), mode="bicubic", align_corners=False
            )
            pos_space = pos_space.permute(0, 2, 3, 1)
        # Hyperbolic rebuild: recompute time after spatial interpolation.
        pos = _space_to_lorentz(self.manifold, pos_space)
        return pos.flatten(1, 2)

    def prepare_tokens_with_masks(
        self, x: Tensor, masks=None
    ) -> Tuple[Tensor, Tuple[int, int]]:
        x = self.patch_embed(x)
        B, H, W, _ = x.shape
        x = x.flatten(1, 2)

        mask_token = self._build_token(self.mask_token_space)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), mask_token.to(x.dtype).unsqueeze(0), x)
            cls_token = self._build_token(self.cls_token_space)
        else:
            cls_token = self._build_token(self.cls_token_space) + 0 * mask_token

        if self.use_pos_embed:
            pos_tokens = self._get_pos_embed(H, W)
            x = self.pos_embed_add(x, pos_tokens)

        if self.n_storage_tokens > 0:
            storage_tokens = self._build_token(self.storage_tokens_space)
        else:
            storage_tokens = torch.empty(
                1,
                0,
                cls_token.shape[-1],
                dtype=cls_token.dtype,
                device=cls_token.device,
            )

        x = torch.cat(
            [
                cls_token.expand(B, -1, -1),
                storage_tokens.expand(B, -1, -1),
                x,
            ],
            dim=1,
        )

        return x, (H, W)

    def forward_features_list(
        self, x_list: List[Tensor], masks_list: List[Tensor]
    ) -> List[Dict[str, Tensor]]:
        x = []
        rope = []
        for t_x, t_masks in zip(x_list, masks_list):
            t2_x, hw_tuple = self.prepare_tokens_with_masks(t_x, t_masks)
            x.append(t2_x)
            rope.append(hw_tuple)
        for _, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = [
                    self._build_rope(H, W, x_i.device) for x_i, (H, W) in zip(x, rope)
                ]
            else:
                rope_sincos = [None for _ in rope]
            x = blk(x, rope_sincos)
        all_x = x
        output = []
        for idx, (x, masks) in enumerate(zip(all_x, masks_list)):
            if self.untie_cls_and_patch_norms or self.untie_global_and_local_cls_norm:
                if self.untie_global_and_local_cls_norm and self.training and idx == 1:
                    x_norm_cls_reg = self.local_cls_norm(
                        x[:, : self.n_storage_tokens + 1]
                    )
                elif self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(x[:, : self.n_storage_tokens + 1])
                else:
                    x_norm_cls_reg = self.norm(x[:, : self.n_storage_tokens + 1])
                x_norm_patch = self.norm(x[:, self.n_storage_tokens + 1 :])
            else:
                x_norm = self.norm(x)
                x_norm_cls_reg = x_norm[:, : self.n_storage_tokens + 1]
                x_norm_patch = x_norm[:, self.n_storage_tokens + 1 :]
            output.append(
                {
                    "x_norm_clstoken": x_norm_cls_reg[:, 0],
                    "x_storage_tokens": x_norm_cls_reg[:, 1:],
                    "x_norm_patchtokens": x_norm_patch,
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(
        self, x: Tensor | List[Tensor], masks: Optional[Tensor] = None
    ) -> List[Dict[str, Tensor]]:
        if isinstance(x, torch.Tensor):
            return self.forward_features_list([x], [masks])[0]
        return self.forward_features_list(x, masks)

    def _get_intermediate_layers_not_chunked(
        self, x: Tensor, n: int = 1
    ) -> List[Tensor]:
        x, (H, W) = self.prepare_tokens_with_masks(x)
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = (
            range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        )
        for i, blk in enumerate(self.blocks):
            rope = (
                self._build_rope(H, W, x.device)
                if self.rope_embed is not None
                else None
            )
            x = blk(x, rope)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(
            blocks_to_take
        ), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        *,
        n: Union[int, Sequence] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs_normed = []
            for out in outputs:
                if self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(out[:, : self.n_storage_tokens + 1])
                    x_norm_patch = self.norm(out[:, self.n_storage_tokens + 1 :])
                    outputs_normed.append(
                        torch.cat((x_norm_cls_reg, x_norm_patch), dim=1)
                    )
                else:
                    outputs_normed.append(self.norm(out))
            outputs = outputs_normed
        class_tokens = [out[:, 0] for out in outputs]
        extra_tokens = [out[:, 1 : self.n_storage_tokens + 1] for out in outputs]
        outputs = [out[:, self.n_storage_tokens + 1 :] for out in outputs]
        if reshape:
            B, _, h, w = x.shape
            patch_h, patch_w = _make_2tuple(self.patch_size)
            outputs = [
                out.reshape(B, h // patch_h, w // patch_w, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
                for out in outputs
            ]
        if not return_class_token and not return_extra_tokens:
            return tuple(outputs)
        if return_class_token and not return_extra_tokens:
            return tuple(zip(outputs, class_tokens))
        if not return_class_token and return_extra_tokens:
            return tuple(zip(outputs, extra_tokens))
        return tuple(zip(outputs, class_tokens, extra_tokens))

    def forward(
        self, *args, is_training: bool = False, **kwargs
    ) -> List[Dict[str, Tensor]] | Tensor:
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        return self.head(ret["x_norm_clstoken"])


def vit_small(
    patch_size=16, embed_dim=384, depth=12, num_heads=6, ffn_ratio=4, **kwargs
):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=embed_dim + num_heads if embed_dim == 384 else embed_dim,
        depth=depth,
        num_heads=num_heads,
        ffn_ratio=ffn_ratio,
        **kwargs,
    )
    return model


def vit_base(
    patch_size=16, embed_dim=768, depth=12, num_heads=12, ffn_ratio=4, **kwargs
):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        ffn_ratio=ffn_ratio,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_so400m(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1152,
        depth=27,
        num_heads=18,
        ffn_ratio=3.777777778,
        **kwargs,
    )
    return model


def vit_huge2(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1280,
        depth=32,
        num_heads=20,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_7b(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=4096,
        depth=40,
        num_heads=32,
        ffn_ratio=3,
        **kwargs,
    )
    return model

if __name__ == "__main__":
    model = vit_base(num_classes=0, device="cpu")
    model.init_weights()
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input, is_training=False)
    print(output.shape)  # Expected output: torch.Size([2, 1000])
