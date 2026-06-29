# LFCS2M 

This folder adds a test script for Synthetic-to-Measured SAR image translation.

## What is included

```text
LFCS2M/
├── translate_test.py              # inference-only entry point
├── configs/lfcs2m_inference.yaml  # default inference config
├── models/lfcs2m.py               # LFCS2M / FDFRM / MIGCA inference modules
├── utils/                         # SAR image I/O and config helpers
├── scripts/run_lfcs2m_translation.sh
├── weights/README.md              # where to put trained weights
├── data/README.md                 # expected input layout
└── results/                       # default output folder
```


- `FDFRM`: frequency-domain feature refinement module with a learnable 2-D spectral mask.
- `MIGCA`: measured-information-guided cross-attention module.

## Recommended placement

Place the folder like this:

```text
LEDS2M/
├── main.py
├── configs/
├── runners/
├── ...
└── LFCS2M_release/
```

Only files inside `LFCS2M_release/` are new.

## Install minimal dependencies

From the repository root:

```bash
pip install -r LFCS2M_release/requirements.txt
```

If the original LEDS2M environment has already been created, the standalone dependencies are usually already available.

## Prepare weights

Put the trained checkpoint here:

```text
LFCS2M_release/weights/lfcs2m.pth
```

Weights are intentionally not included in this folder.

## Prepare test images

For standalone inference, put synthetic SAR images in one directory:

```text
LFCS2M_release/data/synthetic_test/
├── sample_001.png
├── sample_002.png
└── ...
```

Supported image extensions: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`.

## Run standalone LFCS2M inference

```bash
python LFCS2M/translate_test.py \
  --backend standalone \
  --input_dir LFCS2M/data/synthetic_test \
  --output_dir LFCS2M/results/translated \
  --checkpoint LFCS2M/weights/lfcs2m.pth \
  --device cuda:0 \
  --steps 200
```





