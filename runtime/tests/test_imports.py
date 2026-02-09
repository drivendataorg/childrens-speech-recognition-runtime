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
