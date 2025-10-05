import os
import random
import numpy as np

from .indextts2_node import (
    _audio_to_temp_wav,
    _get_tts2_model,
    _resolve_device,
)


def _coerce_float(value, default, clamp=None, random_bounds=None):
    def _maybe_random():
        if not random_bounds:
            base = default if isinstance(default, (int, float)) else 0.0
            low = max(0.0, base * 0.5) if base else 0.0
            high = base * 1.5 + 1.0 if base else 1.0
            return random.uniform(low, high)
        low, high = random_bounds
        return random.uniform(low, high)

    if value is None:
        return default
    if isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, str):
        token = value.strip().lower()
        if not token:
            return default
        if token in {"random", "rand", "randomize"}:
            result = float(_maybe_random())
        else:
            try:
                result = float(token)
            except ValueError:
                return default
    else:
        return default
    if clamp is not None:
        low, high = clamp
        if low is not None:
            result = max(low, result)
        if high is not None:
            result = min(high, result)
    return result


def _coerce_int(value, default, clamp=None, random_bounds=None):
    if value is None:
        result = default
    elif isinstance(value, (int, float)):
        result = int(value)
    elif isinstance(value, str):
        token = value.strip().lower()
        if not token:
            return default
        if token in {"random", "rand", "randomize"}:
            if random_bounds is not None:
                low, high = random_bounds
            else:
                base = default if isinstance(default, (int, float)) else 0
                low = max(0, int(base * 0.5))
                high = max(low + 1, int(base * 1.5) + 1)
            if high <= low:
                high = low + 1
            result = random.randint(int(low), int(high))
        else:
            try:
                result = int(float(token))
            except ValueError:
                return default
    else:
        return default
    if clamp is not None:
        low, high = clamp
        if low is not None:
            result = max(int(low), result)
        if high is not None:
            result = min(int(high), result)
    return result


def _coerce_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"", "none"}:
            return default
        if token in {"true", "1", "yes", "on"}:
            return True
        if token in {"false", "0", "no", "off"}:
            return False
    return bool(value)


class IndexTTS2Advanced:
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
                "use_random_style": ("BOOLEAN", {"default": False}),
                "interval_silence_ms": ("INT", {"default": 200, "min": 0, "max": 12000, "step": 10}),
                "max_text_tokens_per_segment": ("INT", {"default": 120, "min": 0, "max": 2048, "step": 8}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "do_sample": ("BOOLEAN", {"default": True}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 5.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 30, "min": 0, "max": 2048, "step": 1}),
                "repetition_penalty": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                "length_penalty": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 50.0, "step": 0.1}),
                "num_beams": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "max_mel_tokens": ("INT", {"default": 1500, "min": 0, "max": 8192, "step": 8}),
                "typical_sampling": ("BOOLEAN", {"default": False}),
                "typical_mass": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 2000.0, "step": 0.01}),
                "speech_speed": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "use_fp16": ("BOOLEAN", {"default": False}),
                "output_gain": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
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
                   emotion_vector=None,
                   use_random_style: bool = False,
                   interval_silence_ms: int = 200,
                   max_text_tokens_per_segment: int = 120,
                   seed: int = -1,
                   do_sample: bool = True,
                   temperature: float = 0.8,
                   top_p: float = 0.8,
                   top_k: int = 30,
                   repetition_penalty: float = 10.0,
                   length_penalty: float = 0.0,
                   num_beams: int = 3,
                   max_mel_tokens: int = 1500,
                   typical_sampling: bool = False,
                   typical_mass: float = 0.9,
                   speech_speed: float = 1.0,
                   use_fp16: bool = False,
                   output_gain: float = 1.0):

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

        seed_value = None
        if isinstance(seed, (int, np.integer)) and int(seed) >= 0:
            seed_value = int(seed)
            random.seed(seed_value)
            np.random.seed(seed_value)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        ext_root = os.path.dirname(base_dir)
        resolved_model_dir = os.path.join(ext_root, "checkpoints")
        resolved_config = os.path.join(resolved_model_dir, "config.yaml")

        if not os.path.isfile(resolved_config):
            raise FileNotFoundError(f"Config file not found: {resolved_config}")
        if not os.path.isdir(resolved_model_dir):
            raise FileNotFoundError(f"Model directory not found: {resolved_model_dir}")

        resolved_device = _resolve_device("auto")
        use_fp16_flag = _coerce_bool(use_fp16, False)
        tts2 = _get_tts2_model(
            config_path=resolved_config,
            model_dir=resolved_model_dir,
            device=resolved_device,
            use_cuda_kernel=False,
            use_fp16=use_fp16_flag,
        )

        torch_mod = None

        def _ensure_torch():
            nonlocal torch_mod
            if torch_mod is None:
                try:
                    import torch as torch_lib  # type: ignore
                except Exception as exc:
                    raise RuntimeError(f"PyTorch is required for IndexTTS2 Advanced: {exc}")
                torch_mod = torch_lib
            return torch_mod

        seed_info = "random"
        if seed_value is not None:
            torch_lib = _ensure_torch()
            torch_lib.manual_seed(seed_value)
            if torch_lib.cuda.is_available():
                torch_lib.cuda.manual_seed_all(seed_value)
            if hasattr(torch_lib, "xpu") and callable(getattr(torch_lib.xpu, "is_available", None)) and torch_lib.xpu.is_available():
                torch_lib.xpu.manual_seed_all(seed_value)
            if hasattr(torch_lib.backends, "mps") and torch_lib.backends.mps.is_available():
                try:
                    torch_lib.manual_seed(seed_value)
                except Exception:
                    pass
            seed_info = str(seed_value)

        emo_alpha = max(0.0, min(1.0, float(emotion_control_weight)))
        emo_audio_prompt = emo_path if emo_path else prompt_path
        ui_msgs = []
        ui_msgs.append(f"Model precision: {'FP16' if use_fp16_flag else 'FP32'}")

        gain_value = _coerce_float(output_gain, 1.0, clamp=(0.0, 4.0))

        emo_vector_arg = None
        if emotion_vector is not None:
            try:
                vec = [max(0.0, float(v)) for v in list(emotion_vector)[:8]]
                while len(vec) < 8:
                    vec.append(0.0)
                emo_vector_arg = vec
                emo_audio_prompt = prompt_path
                if emo_path is not None:
                    ui_msgs.append("Emotion source: vectors (second audio ignored)")
                else:
                    ui_msgs.append("Emotion source: vectors")
            except Exception:
                emo_vector_arg = None

        if emo_vector_arg is None:
            if emo_path is not None:
                ui_msgs.append("Emotion source: second audio")
            else:
                ui_msgs.append("Emotion source: original audio")

        use_random_style = _coerce_bool(use_random_style, False)
        if use_random_style:
            ui_msgs.append("Emotion source: random preset mix")

        do_sample = _coerce_bool(do_sample, True)
        typical_sampling = _coerce_bool(typical_sampling, False)

        interval_silence_ms = _coerce_int(interval_silence_ms, 200, clamp=(0, 12000))
        max_text_tokens_per_segment = _coerce_int(max_text_tokens_per_segment, 120, clamp=(0, 2048))
        if max_text_tokens_per_segment <= 0:
            max_text_tokens_per_segment = 120

        top_k = _coerce_int(top_k, 30, clamp=(0, 2048))
        num_beams = max(1, _coerce_int(num_beams, 3, clamp=(1, 128)))
        max_mel_tokens = _coerce_int(max_mel_tokens, 1500, clamp=(1, 8192))

        temperature = _coerce_float(temperature, 0.8, clamp=(0.0, 5.0), random_bounds=(0.6, 1.4))
        if temperature < 1e-4:
            temperature = 1e-4
        top_p = _coerce_float(top_p, 0.8, clamp=(0.0, 1.0), random_bounds=(0.5, 0.95))
        repetition_penalty = _coerce_float(repetition_penalty, 10.0, clamp=(0.0, 50.0))
        length_penalty = _coerce_float(length_penalty, 0.0, clamp=(-10.0, 50.0))
        typical_mass = _coerce_float(typical_mass, 0.9, clamp=(0.0, 0.99), random_bounds=(0.5, 0.95))
        if typical_mass <= 0.0:
            typical_mass = 0.9

        speech_speed = _coerce_float(speech_speed, 1.0, clamp=(0.25, 4.0), random_bounds=(0.6, 1.4))

        generation_kwargs = {
            "do_sample": bool(do_sample),
            "top_p": top_p,
            "top_k": top_k,
            "temperature": temperature,
            "length_penalty": float(length_penalty),
            "num_beams": num_beams,
            "repetition_penalty": repetition_penalty,
            "max_mel_tokens": max_mel_tokens,
            "typical_sampling": bool(typical_sampling),
            "typical_mass": typical_mass,
            "speech_speed": speech_speed,
        }
        try:
            result = tts2.infer(
                spk_audio_prompt=prompt_path,
                text=text,
                output_path=None,
                emo_audio_prompt=emo_audio_prompt,
                emo_alpha=emo_alpha,
                emo_vector=emo_vector_arg,
                use_random=bool(use_random_style),
                interval_silence=interval_silence_ms,
                verbose=False,
                max_text_tokens_per_segment=max_text_tokens_per_segment,
                **generation_kwargs,
            )
        finally:
            try:
                if need_cleanup and prompt_path and os.path.exists(prompt_path):
                    os.remove(prompt_path)
                if emo_need_cleanup and emo_path and os.path.exists(emo_path):
                    os.remove(emo_path)
            except Exception:
                pass

        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise RuntimeError("IndexTTS2 returned an unexpected result format")

        sr, wav = result
        torch_lib = _ensure_torch()
        if hasattr(wav, "cpu"):
            wav = wav.cpu().numpy()
        wav = np.asarray(wav)

        if wav.dtype == np.int16:
            wav = wav.astype(np.float32) / 32767.0
        elif wav.dtype != np.float32:
            wav = wav.astype(np.float32)

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

        info_lines = []
        if ui_msgs:
            info_lines.extend(ui_msgs)

        if gain_value != 1.0:
            mono = np.clip(mono * gain_value, -1.0, 1.0)
            info_lines.append(f"Output gain applied: {gain_value:.2f}x")

        waveform = torch_lib.from_numpy(mono[None, None, :].astype(np.float32))

        info_lines.append(f"Seed: {seed_info}")
        if do_sample:
            info_lines.append(f"Sampling: temp={temperature:.2f}, top_p={top_p:.2f}, top_k={top_k}")
        else:
            info_lines.append(f"Beam search: num_beams={num_beams}")
        info_lines.append(f"Repetition penalty={repetition_penalty:.2f}, max_mel_tokens={max_mel_tokens}")
        info_lines.append(f"Speech speed scale={speech_speed:.2f}, interval_silence_ms={interval_silence_ms}")
        if typical_sampling:
            info_lines.append(f"Typical sampling mass={typical_mass:.2f}")

        info_text = "\n".join(info_lines)
        return ({"sample_rate": int(sr), "waveform": waveform}, info_text)



