# LFCS2M 

This folder adds a test script for LFCS2M.

The complete training code and pretrained weights are coming soon.

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

## Data preparation

### Standalone LFCS2M inference

For standalone inference, put synthetic SAR images in one directory:

```text
LFCS2M/data/synthetic_test/
├── sample_001.png
├── sample_002.png
└── ...
```

Supported image extensions: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`.

The script translates each synthetic SAR image into a measured-like SAR image and saves the result under `LFCS2M/results/translated` by default.

### Paired SAMPLE SAR translation task

For the original LEDS2M/BBDM backend or paired LFCS2M training data, prepare the [SAMPLE](https://github.com/benjaminlewis-afrl/SAMPLE_dataset_public) SAR dataset with the aligned directory layout expected by `custom_aligned`.

In the current project configuration, `stage/A` is the condition/reference domain and `stage/B` is the target/ground-truth domain. The existing SAMPLE configuration uses real measured SAR images in `A` and synthetic SAR images in `B`:

```text
SAMPLE_dataset/train/A  # training real/measured reference
SAMPLE_dataset/train/B  # training synthetic target
SAMPLE_dataset/val/A    # validating real/measured reference
SAMPLE_dataset/val/B    # validating synthetic target
SAMPLE_dataset/test/A   # testing real/measured reference
SAMPLE_dataset/test/B   # testing synthetic target
```

Then point the original repository config to the dataset root:

```yaml
data:
  dataset_name: 'SAMPLE'
  dataset_type: 'custom_aligned'
  dataset_config:
    dataset_path: '/path/to/SAMPLE_dataset'
    image_size: 256
    channels: 3
    to_normal: True
    flip: False
```

The default paired-data template is `configs/Template-LBBDM-f4.yaml` in the original repository root. When using `translate.py --backend leds2m`, the same path can also be overridden with `--dataset_path /path/to/SAMPLE_dataset`.

## Run LFCS2M inference

```bash
python LFCS2M/translate_test.py \
  --backend standalone \
  --input_dir LFCS2M/data/synthetic_test \
  --output_dir LFCS2M/results/translated \
  --checkpoint LFCS2M/weights/lfcs2m.pth \
  --device cuda:0 \
  --steps 200
```



