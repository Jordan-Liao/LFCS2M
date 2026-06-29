
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

LFCS2M_MODULE_NAME = "LFCS2M"
FDFRM_MODULE_NAME = "FDFRM"
MIGCA_MODULE_NAME = "MIGCA"


class FDFRM(nn.Module):
    """Frequency Domain Feature Refinement Module.

    The module learns a two-dimensional spectral mask and applies it to the
    Fourier representation of latent features. This follows the paper intuition
    that the synthetic-to-measured SAR gap is structured across frequency bands
    and orientations.
    """

    def __init__(self, channels: int, mask_size: int = 64, alpha: float = 0.02):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if mask_size <= 0:
            raise ValueError("mask_size must be positive")

        initial_mask = torch.ones(1, channels, mask_size, mask_size)
        initial_mask = initial_mask + alpha * torch.randn_like(initial_mask)
        self.spectral_mask = nn.Parameter(initial_mask)
        self.residual_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"FDFRM expects BCHW input, got shape={tuple(x.shape)}")
        _, _, height, width = x.shape

        # rFFT keeps only non-redundant frequencies in the last dimension.
        freq = torch.fft.rfft2(x, norm="ortho")
        mask = F.interpolate(
            self.spectral_mask,
            size=(height, width // 2 + 1),
            mode="bilinear",
            align_corners=False,
        )
        refined = torch.fft.irfft2(freq * mask.to(dtype=freq.dtype), s=(height, width), norm="ortho")
        return x + self.residual_scale * refined


class MIGCA(nn.Module):
    """Measured Information Guided Cross Attention.

    At test time a measured reference is unavailable, so the guidance tensor is
    produced from the current latent trajectory. During a supervised training
    setup, the same module can receive a measured latent as `guide`.
    """

    def __init__(self, channels: int, heads: int = 4, max_tokens: int = 1024):
        super().__init__()
        if channels % heads != 0:
            raise ValueError("channels must be divisible by heads")
        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        self.max_tokens = max_tokens

        self.to_q = nn.Conv2d(channels, channels, kernel_size=1)
        self.to_k = nn.Conv2d(channels, channels, kernel_size=1)
        self.to_v = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.scale = self.head_dim ** -0.5

    def _flatten_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = x.view(b, self.heads, self.head_dim, h * w)
        return x.transpose(-2, -1)  # B, heads, tokens, head_dim

    def _maybe_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        _, _, height, width = x.shape
        tokens = height * width
        if tokens <= self.max_tokens:
            return x, (height, width)
        scale = (self.max_tokens / float(tokens)) ** 0.5
        pooled_h = max(8, int(height * scale))
        pooled_w = max(8, int(width * scale))
        return F.adaptive_avg_pool2d(x, (pooled_h, pooled_w)), (height, width)

    def forward(self, st: torch.Tensor, guide: Optional[torch.Tensor] = None) -> torch.Tensor:
        if guide is None:
            guide = st
        if st.shape != guide.shape:
            guide = F.interpolate(guide, size=st.shape[-2:], mode="bilinear", align_corners=False)

        st_pooled, original_hw = self._maybe_pool(st)
        guide_pooled = F.adaptive_avg_pool2d(guide, st_pooled.shape[-2:])

        q = self._flatten_heads(self.to_q(self.norm(st_pooled)))
        k = self._flatten_heads(self.to_k(self.norm(guide_pooled)))
        v = self._flatten_heads(self.to_v(guide_pooled))

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        out = torch.matmul(attn, v)  # B, heads, tokens, head_dim
        b, _, tokens, _ = out.shape
        h, w = st_pooled.shape[-2:]
        out = out.transpose(-2, -1).contiguous().view(b, self.channels, h, w)
        out = self.proj(out)

        if (h, w) != original_hw:
            out = F.interpolate(out, size=original_hw, mode="bilinear", align_corners=False)
        return st + out


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(num_groups=min(8, channels), num_channels=channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(num_groups=min(8, channels), num_channels=channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class EncoderWithFDFRM(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 64, latent_channels: int = 64,
                 mask_size: int = 64, use_fdf_rm: bool = True):
        super().__init__()
        self.stem = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.fdf_rm = FDFRM(base_channels, mask_size=mask_size) if use_fdf_rm else nn.Identity()
        self.body = nn.Sequential(
            ResidualBlock(base_channels),
            nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1),
            ResidualBlock(base_channels * 2),
            nn.Conv2d(base_channels * 2, latent_channels, 4, stride=2, padding=1),
            ResidualBlock(latent_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.fdf_rm(x)
        return self.body(x)


class Decoder(nn.Module):
    def __init__(self, out_channels: int = 3, base_channels: int = 64, latent_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            ResidualBlock(latent_channels),
            nn.ConvTranspose2d(latent_channels, base_channels * 2, 4, stride=2, padding=1),
            ResidualBlock(base_channels * 2),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            ResidualBlock(base_channels),
            nn.Conv2d(base_channels, out_channels, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class TinyUNetDenoiser(nn.Module):
    """Compact denoiser used by the LFCS2M reverse bridge at inference."""

    def __init__(self, channels: int, base_channels: int = 64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, channels),
            nn.SiLU(inplace=True),
            nn.Linear(channels, channels),
        )
        self.in_block = nn.Sequential(nn.Conv2d(channels, base_channels, 3, padding=1), ResidualBlock(base_channels))
        self.down = nn.Sequential(nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1), ResidualBlock(base_channels * 2))
        self.mid = nn.Sequential(ResidualBlock(base_channels * 2), ResidualBlock(base_channels * 2))
        self.up = nn.Sequential(nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1), ResidualBlock(base_channels))
        self.out = nn.Conv2d(base_channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        if timestep.ndim == 0:
            timestep = timestep[None]
        timestep = timestep.to(device=x.device, dtype=x.dtype).view(-1, 1)
        while timestep.shape[0] < x.shape[0]:
            timestep = timestep.repeat(2, 1)
        timestep = timestep[: x.shape[0]]
        emb = self.time_mlp(timestep).view(x.shape[0], x.shape[1], 1, 1)
        x = x + emb
        h1 = self.in_block(x)
        h2 = self.down(h1)
        h2 = self.mid(h2)
        out = self.up(h2)
        out = out + h1
        return self.out(out)


@dataclass
class LFCS2MConfig:
    image_size: int = 256
    in_channels: int = 3
    base_channels: int = 64
    latent_channels: int = 64
    spectral_mask_size: int = 64
    attention_heads: int = 4
    use_fdf_rm: bool = True
    use_migca: bool = True


class LFCS2M(nn.Module):
    """Inference-only LFCS2M wrapper for high-fidelity SAR translation."""

    def __init__(self, config: Optional[LFCS2MConfig] = None, **kwargs):
        super().__init__()
        if config is None:
            config = LFCS2MConfig(**kwargs)
        self.config = config
        self.encoder_fef = EncoderWithFDFRM(
            in_channels=config.in_channels,
            base_channels=config.base_channels,
            latent_channels=config.latent_channels,
            mask_size=config.spectral_mask_size,
            use_fdf_rm=config.use_fdf_rm,
        )
        self.migca = MIGCA(config.latent_channels, heads=config.attention_heads) if config.use_migca else nn.Identity()
        self.denoiser = TinyUNetDenoiser(config.latent_channels, base_channels=config.base_channels)
        self.decoder = Decoder(out_channels=config.in_channels, base_channels=config.base_channels, latent_channels=config.latent_channels)

    def encode(self, synthetic: torch.Tensor) -> torch.Tensor:
        return self.encoder_fef(synthetic)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    @torch.no_grad()
    def sample(self, synthetic: torch.Tensor, steps: int = 200, clip_denoised: bool = True) -> torch.Tensor:
        """Translate synthetic SAR images into measured-like SAR images.

        Parameters
        ----------
        synthetic:
            Tensor in range [-1, 1] with shape [B, C, H, W].
        steps:
            Number of reverse bridge steps. Use the value used for your checkpoint.
        clip_denoised:
            Whether to clip the decoded result to [-1, 1].
        """
        if steps <= 0:
            raise ValueError("steps must be positive")
        st = self.encode(synthetic)
        guide = st
        for idx in reversed(range(steps)):
            t_scalar = torch.full((st.shape[0],), float(idx + 1) / float(steps), device=st.device, dtype=st.dtype)
            guided = self.migca(st, guide) if isinstance(self.migca, MIGCA) else st
            predicted_noise = self.denoiser(guided, t_scalar)
            # Conservative deterministic reverse update. The trained checkpoint
            # determines the actual measured-domain direction.
            step_size = 1.0 / float(steps)
            st = st - step_size * predicted_noise
        out = self.decode(st)
        if clip_denoised:
            out = out.clamp(-1.0, 1.0)
        return out


def _strip_prefix_if_present(state_dict: Mapping[str, torch.Tensor], prefixes: Iterable[str]) -> OrderedDict:
    cleaned = OrderedDict()
    for key, value in state_dict.items():
        new_key = key
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def _select_state_dict(checkpoint: Mapping) -> Mapping[str, torch.Tensor]:
    for key in ("state_dict", "model", "net", "ema_model", "model_state_dict"):
        if key in checkpoint and isinstance(checkpoint[key], Mapping):
            return checkpoint[key]
    return checkpoint


def load_lfcs2m_checkpoint(model: nn.Module, checkpoint_path: str | Path, strict: bool = False,
                           map_location: str | torch.device = "cpu") -> Dict[str, Tuple[str, ...]]:
    """Load a trained LFCS2M checkpoint with flexible key handling."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(str(path), map_location=map_location)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")
    state_dict = _select_state_dict(checkpoint)
    state_dict = _strip_prefix_if_present(state_dict, prefixes=("module.", "model.", "net.", "lfcs2m."))
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    return {"missing_keys": tuple(missing), "unexpected_keys": tuple(unexpected)}
