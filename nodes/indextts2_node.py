import gc
import os
import sys
import tempfile
import threading
from functools import wraps
from typing import Any, Dict, Tuple

import numpy as np

#simple in-memory cache for loaded models to avoid re-initializing weights
_MODEL_CACHE: Dict[Tuple[str, str, str, bool, bool], Any] = {}
_CACHE_LOCK = threading.RLock()
_UNLOAD_HOOK_INSTALLED = False

def _resolve_device(device: str):
    try:
        import torch
    except Exception:
        return "cpu"

    if device and device not in ("auto", ""):
        return device
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch, "xpu") and getattr(torch.xpu, "is_available", lambda: False)():
        return "xpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def _get_tts2_model(config_path: str,
                    model_dir: str,
                    device: str,
                    use_cuda_kernel: bool,
                    use_fp16: bool):
    _install_unload_hook()

    key = (os.path.abspath(config_path), os.path.abspath(model_dir), device, bool(use_cuda_kernel), bool(use_fp16))

    with _CACHE_LOCK:
        cached_model = _MODEL_CACHE.get(key)
    if cached_model is not None:
        return cached_model

    base_dir = os.path.dirname(os.path.abspath(__file__))
    ext_root = os.path.dirname(base_dir)
    if ext_root not in sys.path:
        sys.path.insert(0, ext_root)

    #quiet down transformers advisory warnings (e.g., GenerationMixin notice)
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
    except Exception:
        pass

    from indextts.infer_v2 import IndexTTS2

    eff_fp16 = use_fp16 and device.startswith("cuda")

    model = IndexTTS2(
        cfg_path=config_path,
        model_dir=model_dir,
        use_fp16=eff_fp16,
        device=device,
        use_cuda_kernel=use_cuda_kernel,
        use_deepspeed=False,
    )
    with _CACHE_LOCK:
        existing = _MODEL_CACHE.get(key)
        if existing is None:
            _MODEL_CACHE[key] = model
            cached_model = model
        else:
            cached_model = existing
    return cached_model




def _flush_device_caches():
    try:
        import torch
    except Exception:
        torch = None

    if torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
        try:
            if hasattr(torch, "xpu") and getattr(torch.xpu, "is_available", lambda: False)():
                torch.xpu.empty_cache()
        except Exception:
            pass
        try:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass
        try:
            if hasattr(torch, "npu") and getattr(torch.npu, "is_available", lambda: False)():
                torch.npu.empty_cache()
        except Exception:
            pass
        try:
            if hasattr(torch, "mlu") and getattr(torch.mlu, "is_available", lambda: False)():
                torch.mlu.empty_cache()
        except Exception:
            pass
    gc.collect()


def _teardown_model(model):
    try:
        import torch
    except Exception:
        torch = None

    module_attrs = [
        "gpt",
        "semantic_model",
        "semantic_codec",
        "s2mel",
        "campplus_model",
        "bigvgan",
        "qwen_emo",
    ]
    for attr in module_attrs:
        comp = getattr(model, attr, None)
        if comp is None:
            continue
        if torch is not None and hasattr(comp, "to"):
            try:
                comp.to("cpu")
            except Exception:
                pass
        try:
            delattr(model, attr)
        except Exception:
            setattr(model, attr, None)

    tensor_attrs = [
        "semantic_mean",
        "semantic_std",
    ]
    for attr in tensor_attrs:
        value = getattr(model, attr, None)
        if value is None:
            continue
        if torch is not None and hasattr(value, "detach"):
            try:
                value = value.detach().cpu()
            except Exception:
                pass
        setattr(model, attr, None)

    for attr in ("emo_matrix", "spk_matrix"):
        if hasattr(model, attr):
            setattr(model, attr, None)

    cache_attrs = [
        "cache_spk_cond",
        "cache_s2mel_style",
        "cache_s2mel_prompt",
        "cache_spk_audio_prompt",
        "cache_emo_cond",
        "cache_emo_audio_prompt",
        "cache_mel",
    ]
    for attr in cache_attrs:
        if hasattr(model, attr):
            setattr(model, attr, None)

    for attr in ("extract_features", "normalizer", "tokenizer", "mel_fn"):
        if hasattr(model, attr):
            setattr(model, attr, None)


def _dispose_cached_models() -> bool:
    with _CACHE_LOCK:
        if not _MODEL_CACHE:
            return False
        cached_items = list(_MODEL_CACHE.items())
        _MODEL_CACHE.clear()

    for _, model in cached_items:
        try:
            _teardown_model(model)
        except Exception:
            pass

    _flush_device_caches()
    return True


def unload_cached_models() -> bool:
    """Expose manual cache invalidation for other extensions."""
    return _dispose_cached_models()


def _install_unload_hook():
    global _UNLOAD_HOOK_INSTALLED
    if _UNLOAD_HOOK_INSTALLED:
        return
    try:
        import comfy.model_management as mm
    except Exception:
        return
    if getattr(mm.unload_all_models, "_indextts2_hook", False):
        _UNLOAD_HOOK_INSTALLED = True
        return

    original = mm.unload_all_models

    @wraps(original)
    def wrapper(*args, **kwargs):
        _dispose_cached_models()
        return original(*args, **kwargs)

    wrapper._indextts2_hook = True
    mm.unload_all_models = wrapper
    _UNLOAD_HOOK_INSTALLED = True


def _audio_to_temp_wav(audio: Any) -> Tuple[str, int, bool]:

    sr = None
    data = None

    if isinstance(audio, str) and os.path.exists(audio):
        return audio, 0, False  # use existing path, no cleanup

    if isinstance(audio, (tuple, list)):
        cand_ints = [x for x in audio if isinstance(x, (int, np.integer))]
        cand_arrays = [x for x in audio if hasattr(x, "shape")]
        if len(cand_ints) >= 1 and len(cand_arrays) >= 1:
            sr = int(cand_ints[0])
            data = cand_arrays[0]
        elif len(audio) == 2:
            a, b = audio
            if isinstance(a, (int, np.integer)) and hasattr(b, "shape"):
                sr, data = int(a), b
            elif isinstance(b, (int, np.integer)) and hasattr(a, "shape"):
                sr, data = int(b), a

    if sr is None and isinstance(audio, dict):
        sr = audio.get("sample_rate") or audio.get("sr") or audio.get("rate")
        for key in ("waveform", "samples", "audio", "data"):
            if key in audio:
                data = audio[key]
                break

    if sr is None or data is None:
        raise ValueError("Invalid AUDIO input. Expected (sample_rate:int, numpy_array)")

    if hasattr(data, "cpu"):
        data = data.cpu().numpy()
    wav = np.asarray(data)

    if wav.ndim == 1:
        wav = wav[None, :]  # (1, N)
    elif wav.ndim == 2:
        ch_dim = 0 if wav.shape[0] <= 8 and wav.shape[0] <= wav.shape[1] else 1 if wav.shape[1] <= 8 else 0
        if ch_dim == 1:
            wav = np.transpose(wav, (1, 0))  # (N, C) -> (C, N)
    elif wav.ndim >= 3:
        sizes = list(wav.shape)
        sample_axis = int(np.argmax(sizes))
        axes = [i for i in range(wav.ndim) if i != sample_axis] + [sample_axis]
        wav = np.transpose(wav, axes)
        c = int(np.prod(wav.shape[:-1]))
        wav = np.reshape(wav, (c, wav.shape[-1]))
    else:
        raise ValueError("AUDIO array must be 1D or 2D (samples[, channels])")

    if np.issubdtype(wav.dtype, np.integer):
        info = np.iinfo(wav.dtype)
        denom = float(max(abs(info.min), abs(info.max))) or 32767.0
        wav = wav.astype(np.float32) / denom
    else:
        wav = np.clip(wav.astype(np.float32), -1.0, 1.0)

    fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="indextts2_prompt_")
    os.close(fd)
    _save_wav(tmp_path, wav, int(sr))
    return tmp_path, int(sr), True

def _save_wav(path: str, wav_cn: np.ndarray, sr: int):
    """Save numpy waveform to WAV PCM16 without requiring torchaudio.
    Expects wav_cn as (channels, samples) float32 in [-1, 1].
    """
    wav_cn = np.clip(wav_cn, -1.0, 1.0)
    pcm = (wav_cn * 32767.0).astype(np.int16)

    try:
        import soundfile as sf  
        if pcm.ndim == 1:
            sf.write(path, pcm, sr, subtype="PCM_16")
        else:
            sf.write(path, np.transpose(pcm, (1, 0)), sr, subtype="PCM_16")
        return
    except Exception:
        pass

    import wave
    import contextlib

    if pcm.ndim == 1:
        n_channels = 1
        interleaved = pcm.tobytes()
        n_frames = pcm.shape[0]
    else:
        n_channels = int(pcm.shape[0])
        n_frames = int(pcm.shape[1])
        interleaved = np.transpose(pcm, (1, 0)).tobytes()

    with contextlib.closing(wave.open(path, "wb")) as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(int(sr))
        wf.writeframes(interleaved)


_install_unload_hook()

class IndexTTS2Simple:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "text": ("STRING", {"multiline": True}),
                "emotion_control_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "emotion_audio": ("AUDIO",),
                "emotion_vector": ("EMOTION_VECTOR",),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    FUNCTION = "synthesize"
    CATEGORY = "Audio/IndexTTS"

    def synthesize(self,
                   audio,
                   text: str,
                   emotion_control_weight: float,
                   emotion_audio=None,
                   emotion_vector=None):

        if not isinstance(text, str) or len(text.strip()) == 0:
            raise ValueError("Text is empty. Please provide text to synthesize.")

        prompt_path, _, need_cleanup = _audio_to_temp_wav(audio)
        emo_path = None
        emo_need_cleanup = False
        if emotion_audio is not None:
            try:
                emo_path, _, emo_need_cleanup = _audio_to_temp_wav(emotion_audio)
            except Exception:
                emo_path, emo_need_cleanup = None, False

        base_dir = os.path.dirname(os.path.abspath(__file__))
        ext_root = os.path.dirname(base_dir)
        resolved_model_dir = os.path.join(ext_root, "checkpoints")
        resolved_config = os.path.join(resolved_model_dir, "config.yaml")

        if not os.path.isfile(resolved_config):
            raise FileNotFoundError(f"Config file not found: {resolved_config}")
        if not os.path.isdir(resolved_model_dir):
            raise FileNotFoundError(f"Model directory not found: {resolved_model_dir}")

        resolved_device = _resolve_device("auto")
        tts2 = _get_tts2_model(
            config_path=resolved_config,
            model_dir=resolved_model_dir,
            device=resolved_device,
            use_cuda_kernel=False,
            use_fp16=True,
        )

        emo_alpha = max(0.0, min(1.0, float(emotion_control_weight)))
        emo_vector = None
        ui_msgs = []
        if emotion_vector is not None:
            try:
                vec = list(emotion_vector)
                vec = [max(0.0, float(v)) for v in vec][:8]
                while len(vec) < 8:
                    vec.append(0.0)
                emo_vector = vec
                emo_audio_prompt = prompt_path  
                if emo_path is not None:
                    ui_msgs.append("Emotion source: vectors (second audio ignored)")
                else:
                    ui_msgs.append("Emotion source: vectors")
            except Exception:
                emo_vector = None
                emo_audio_prompt = emo_path if emo_path else prompt_path
        else:
            emo_audio_prompt = emo_path if emo_path else prompt_path
            if emo_path is not None:
                ui_msgs.append("Emotion source: second audio")
            else:
                ui_msgs.append("Emotion source: original audio")

        try:
            result = tts2.infer(
                spk_audio_prompt=prompt_path,
                text=text,
                output_path=None,
                emo_audio_prompt=emo_audio_prompt,
                emo_alpha=emo_alpha,
                emo_vector=emo_vector,
                verbose=False,
                interval_silence=200,
            )
        finally:
            #clean up temp files
            try:
                if need_cleanup and prompt_path and os.path.exists(prompt_path):
                    os.remove(prompt_path)
                if emo_need_cleanup and emo_path and os.path.exists(emo_path):
                    os.remove(emo_path)
            except Exception:
                pass

        if not isinstance(result, (tuple, list)) or len(result) != 2:
            #defensive: if the upstream API changes unexpectedly
            raise RuntimeError("IndexTTS2 returned an unexpected result format")

        sr, wav = result
        if hasattr(wav, "cpu"):
            wav = wav.cpu().numpy()
        wav = np.asarray(wav)

        if wav.dtype == np.int16:
            wav = (wav.astype(np.float32) / 32767.0)
        elif wav.dtype != np.float32:
            wav = wav.astype(np.float32)

        try:
            import torch
        except Exception as e:
            raise RuntimeError(f"PyTorch is required to return AUDIO to ComfyUI: {e}")

        mono = wav
        if mono.ndim == 2:
            if mono.shape[0] <= 8 and mono.shape[1] > mono.shape[0]:
                mono = mono.mean(axis=0)
            else:
                mono = mono.mean(axis=-1)
        elif mono.ndim > 2:
            mono = mono.reshape(-1, mono.shape[-1]).mean(axis=0)
        if mono.ndim != 1:
            mono = mono.flatten()

        waveform = torch.from_numpy(mono[None, None, :].astype(np.float32))  #(B=1, C=1, N)
        info_text = "\n".join(ui_msgs) if ui_msgs else ""
        return ({"sample_rate": int(sr), "waveform": waveform}, info_text)
