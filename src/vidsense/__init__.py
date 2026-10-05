"""VidSense: ask questions about a video and get answers with timestamps."""

import os as _os

# Runtime defaults, set before torch / transformers are imported. Users can override any of them.
for _key, _value in {
    "TOKENIZERS_PARALLELISM": "false",  # avoid fork warnings from Hugging Face tokenizers
    "PYTORCH_ENABLE_MPS_FALLBACK": "1",  # run ops the Apple GPU lacks on the CPU instead of failing
    # openai/clip-vit-base-patch32 ships only pytorch_model.bin; without this, transformers
    # downloads a converted 600 MB safetensors copy in a background thread on first load.
    "DISABLE_SAFETENSORS_CONVERSION": "1",
}.items():
    _os.environ.setdefault(_key, _value)

__version__ = "0.1.0"
