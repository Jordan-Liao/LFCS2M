# Weights

Place your trained LFCS2M checkpoint here, for example:

```text
LFCS2M_release/weights/lfcs2m.pth
```

The inference script accepts common PyTorch checkpoint formats:

- raw `state_dict`
- `{ "state_dict": ... }`
- `{ "model": ... }`
- `{ "net": ... }`
- `{ "ema_model": ... }`

For checkpoints trained by the original LEDS2M repository, prefer:

```bash
python LFCS2M_release/translate_test.py --backend leds2m ...
```
