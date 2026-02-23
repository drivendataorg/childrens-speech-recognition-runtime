# Changelog


## 2026-02-23

- Removed vLLM dependency from the competition runtime. vLLM is incompatible with the runtime CUDA (12.6) / PyTorch (2.9.0) stack and cannot be reliably supported.

## 2026-02-09

- Add qwen-asr package for Qwen3-ASR model support [#1](https://github.com/drivendataorg/childrens-speech-recognition-runtime/pull/1)
    - Added `qwen-asr` package (v0.0.6) for Qwen3-ASR model support
    - Upgraded `transformers` from 4.53.3 to 4.57.6 (required by qwen-asr model classes)
    - Upgraded `tokenizers` from 0.21.4 to 0.22.2 (required by transformers 4.57.6)
    - Added `override-dependencies` for transformers to resolve NeMo compatibility constraint
