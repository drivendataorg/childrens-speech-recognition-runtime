def test_pytorch():
    import torch

    assert torch.version.cuda is not None, torch.version.cuda


def test_torchaudio():
    import torchaudio


def test_whisper():
    # https://huggingface.co/openai/whisper-large-v3#usage
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


def test_canary_qwen():
    # https://huggingface.co/nvidia/canary-qwen-2.5b#how-to-use-this-model
    from nemo.collections.speechlm2.models import SALM


def test_granite_speech():
    # https://huggingface.co/ibm-granite/granite-speech-3.3-8b#usage-with-transformers
    import torch
    import torchaudio
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq


def test_phi_4_multimodal():
    # https://huggingface.co/microsoft/Phi-4-multimodal-instruct#loading-the-model-locally
    import soundfile as sf
    from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig


def test_parakeet_tdt():
    # https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2#how-to-use-this-model
    import nemo.collections.asr as nemo_asr


def test_wav2vec():
    # https://huggingface.co/facebook/wav2vec2-base-960h#usage
    from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


def test_qwen3_asr():
    # https://huggingface.co/Qwen/Qwen3-ASR-1.7B
    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRConfig
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRForConditionalGeneration
    from qwen_asr.core.transformers_backend.processing_qwen3_asr import Qwen3ASRProcessor


def test_qwen3_asr_vllm_inference():
    """End-to-end inference test using qwen-asr with the vLLM backend."""
    from pathlib import Path

    import numpy as np
    import torch

    from qwen_asr import Qwen3ASRModel

    # Prefer a real audio file when available; fall back to random noise.
    audio_rel = Path("data-demo") / "word" / "audio" / "U_a61a29f276533ee2.flac"
    candidates = [
        Path(__file__).resolve().parents[2] / audio_rel,   # local checkout
        Path(__file__).resolve().parents[1] / audio_rel,   # /code_execution layout
        Path("/code_execution/data/audio/U_a61a29f276533ee2.flac"),  # bind-mounted data
    ]
    audio_path = next((p for p in candidates if p.exists()), None)

    if audio_path is not None:
        audio = str(audio_path)
    else:
        audio = (np.random.randn(16000).astype(np.float32), 16000)

    model = Qwen3ASRModel.LLM(
        model="Qwen/Qwen3-ASR-0.6B",
        gpu_memory_utilization=0.8,
        max_model_len=8192,
        max_inference_batch_size=1,
        max_new_tokens=256,
    )
    results = model.transcribe(audio=audio, language="English")
    text = results[0].text
    assert isinstance(text, str), f"Unexpected type: {type(text)}"


def test_tensorflow():
    import tensorflow as tf
    import keras
    
    assert tf.test.is_built_with_gpu_support()
