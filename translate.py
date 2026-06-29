#!/usr/bin/env python3
"""Inference-only test script for LFCS2M SAR image translation.

Two backends are supported:

1. `standalone`: uses the self-contained LFCS2M / FDFRM / MIGCA modules in this
   folder and translates every image in `--input_dir`.
2. `leds2m`: calls the original repository's `main.py --sample_to_eval` with a
   temporary config. This is useful when your checkpoint was trained by the
   original LEDS2M codebase.

No training is performed by this script.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import yaml
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from models import LFCS2M, FDFRM, MIGCA, load_lfcs2m_checkpoint  # noqa: E402
from models.lfcs2m import LFCS2MConfig  # noqa: E402
from utils import list_images, load_sar_image, save_sar_image, load_yaml_config, merge_cli_overrides  # noqa: E402

PAPER_MODULES = {
    "model": "LFCS2M",
    "frequency_module": "FDFRM",
    "attention_module": "MIGCA",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LFCS2M inference-only SAR translation test")
    parser.add_argument("--backend", choices=["standalone", "leds2m"], default="standalone",
                        help="Use standalone LFCS2M modules or call the original LEDS2M repository backend.")

    # Common options
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save translated SAR images.")
    parser.add_argument("--steps", type=int, default=None, help="Reverse diffusion / bridge sampling steps.")

    # Standalone backend options
    parser.add_argument("--config_file", type=str, default=str(THIS_DIR / "configs" / "lfcs2m_inference.yaml"),
                        help="Standalone LFCS2M inference YAML config.")
    parser.add_argument("--input_dir", type=str, default=None, help="Directory containing synthetic SAR images.")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--normalization", choices=["linear", "sar_mean_clip"], default=None,
                        help="Image normalization. Overrides YAML io.normalization when set.")
    parser.add_argument("--strict", action="store_true", help="Use strict checkpoint loading for standalone backend.")
    parser.add_argument("--allow_untrained_demo", action="store_true",
                        help="Allow a smoke-test run without a checkpoint. Not for high-fidelity results.")

    # Original LEDS2M repository backend options
    parser.add_argument("--repo_root", type=str, default=str(THIS_DIR.parent),
                        help="Root of the original LEDS2M repository.")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Dataset path used by the original repository backend.")
    parser.add_argument("--config", type=str, default="configs/Template-LEDS2M-f4.yaml",
                        help="Original LEDS2M config path, relative to repo root or absolute.")
    parser.add_argument("--gpu_ids", type=str, default="0", help="GPU ids passed to original main.py; use -1 for CPU.")
    parser.add_argument("--sample_num", type=int, default=1, help="Number of samples per test image for original backend.")
    return parser.parse_args()


def _resolve_path(path: Optional[str], base: Path = THIS_DIR) -> Optional[Path]:
    if path is None:
        return None
    p = Path(path)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def build_lfcs2m_from_config(config: Dict) -> LFCS2M:
    model_cfg = dict(config.get("model", {}))
    # Drop display-only or unknown keys.
    model_cfg.pop("name", None)
    config_obj = LFCS2MConfig(
        image_size=int(model_cfg.get("image_size", 256)),
        in_channels=int(model_cfg.get("in_channels", 3)),
        base_channels=int(model_cfg.get("base_channels", 64)),
        latent_channels=int(model_cfg.get("latent_channels", 64)),
        spectral_mask_size=int(model_cfg.get("spectral_mask_size", 64)),
        attention_heads=int(model_cfg.get("attention_heads", 4)),
        use_fdf_rm=bool(model_cfg.get("use_fdf_rm", True)),
        use_migca=bool(model_cfg.get("use_migca", True)),
    )
    return LFCS2M(config_obj)


def run_standalone(args: argparse.Namespace) -> None:
    cfg = load_yaml_config(args.config_file)
    cfg = merge_cli_overrides(
        cfg,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        steps=args.steps,
    )

    io_cfg = cfg.get("io", {})
    sampling_cfg = cfg.get("sampling", {})
    model_cfg = cfg.get("model", {})

    input_dir = _resolve_path(io_cfg.get("input_dir"), base=Path.cwd())
    output_dir = _resolve_path(io_cfg.get("output_dir"), base=Path.cwd())
    checkpoint = _resolve_path(io_cfg.get("checkpoint"), base=Path.cwd())
    normalization = args.normalization or io_cfg.get("normalization", "linear")
    steps = int(sampling_cfg.get("steps", 200))
    clip_denoised = bool(sampling_cfg.get("clip_denoised", True))
    image_size = int(model_cfg.get("image_size", 256))

    if input_dir is None:
        raise ValueError("--input_dir or io.input_dir must be specified for standalone backend")
    if output_dir is None:
        raise ValueError("--output_dir or io.output_dir must be specified for standalone backend")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = build_lfcs2m_from_config(cfg).to(device)
    model.eval()

    load_report = {"missing_keys": (), "unexpected_keys": ()}
    if checkpoint is not None and checkpoint.exists():
        load_report = load_lfcs2m_checkpoint(model, checkpoint, strict=args.strict, map_location=device)
    elif not args.allow_untrained_demo:
        raise FileNotFoundError(
            "Trained checkpoint is required for high-fidelity LFCS2M translation. "
            "Pass --checkpoint path/to/lfcs2m.pth, or use --allow_untrained_demo only for a smoke test."
        )

    image_paths = list_images(input_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in: {input_dir}")

    print(f"[LFCS2M] backend=standalone modules={PAPER_MODULES}")
    print(f"[LFCS2M] checkpoint={checkpoint if checkpoint else 'UNTRAINED_SMOKE_TEST'}")
    print(f"[LFCS2M] input_dir={input_dir}")
    print(f"[LFCS2M] output_dir={output_dir}")
    print(f"[LFCS2M] steps={steps}, normalization={normalization}, device={device}")

    with torch.no_grad():
        for image_path in tqdm(image_paths, desc="LFCS2M translation"):
            tensor = load_sar_image(image_path, image_size=image_size, normalization=normalization)
            tensor = tensor.unsqueeze(0).to(device)
            translated = model.sample(tensor, steps=steps, clip_denoised=clip_denoised)
            rel_path = image_path.relative_to(input_dir)
            out_name = rel_path.with_suffix("").as_posix().replace("/", "__") + "_LFCS2M.png"
            save_sar_image(translated[0], output_dir / out_name)

    manifest = {
        "backend": "standalone",
        "paper_modules": PAPER_MODULES,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "num_images": len(image_paths),
        "steps": steps,
        "normalization": normalization,
        "checkpoint_load_report": {
            "missing_keys": list(load_report.get("missing_keys", ())),
            "unexpected_keys": list(load_report.get("unexpected_keys", ())),
        },
    }
    with (output_dir / "lfcs2m_inference_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[LFCS2M] Done. Translated {len(image_paths)} images.")


def _load_original_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def run_original_leds2m_backend(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    main_py = repo_root / "main.py"
    if not main_py.exists():
        raise FileNotFoundError(f"Could not find original main.py under repo root: {repo_root}")

    checkpoint = _resolve_path(args.checkpoint, base=Path.cwd())
    if checkpoint is None or not checkpoint.exists():
        raise FileNotFoundError("--checkpoint is required for the original LEDS2M backend")

    original_config = Path(args.config)
    if not original_config.is_absolute():
        original_config = repo_root / original_config
    if not original_config.exists():
        raise FileNotFoundError(f"Original config not found: {original_config}")

    output_dir = _resolve_path(args.output_dir or str(THIS_DIR / "results" / "leds2m_backend"), base=Path.cwd())
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_original_yaml(original_config)
    cfg.setdefault("testing", {})["sample_num"] = int(args.sample_num)
    cfg.setdefault("model", {})["model_name"] = "LFCS2M-inference"
    cfg["model"]["model_load_path"] = str(checkpoint)
    if args.dataset_path is not None:
        cfg.setdefault("data", {}).setdefault("dataset_config", {})["dataset_path"] = args.dataset_path

    runtime_dir = THIS_DIR / "_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_config = runtime_dir / "lfcs2m_runtime_from_leds2m.yaml"
    with runtime_config.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, sort_keys=False, allow_unicode=True)

    command = [
        sys.executable,
        str(main_py),
        "--config", str(runtime_config),
        "--sample_to_eval",
        "--gpu_ids", str(args.gpu_ids),
        "--resume_model", str(checkpoint),
        "--result_path", str(output_dir),
    ]

    print(f"[LFCS2M] backend=leds2m modules={PAPER_MODULES}")
    print("[LFCS2M] Running original repository inference command:")
    print(" ".join(command))
    subprocess.run(command, cwd=str(repo_root), check=True)


def main() -> None:
    args = parse_args()
    if args.backend == "standalone":
        run_standalone(args)
    elif args.backend == "leds2m":
        run_original_leds2m_backend(args)
    else:
        raise ValueError(f"Unknown backend: {args.backend}")


if __name__ == "__main__":
    main()
