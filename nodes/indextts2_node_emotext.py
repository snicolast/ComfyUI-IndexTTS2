import os
from typing import Tuple


_QWEN_CACHE = None


def _get_qwen(model_dir: str):
    global _QWEN_CACHE
    if _QWEN_CACHE is not None:
        return _QWEN_CACHE

    try:
        from modelscope import AutoModelForCausalLM  # noqa: F401
    except Exception:
        raise ImportError("modelscope is required for Emotion From Text. Install with: pip install modelscope")

    import sys
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ext_root = os.path.dirname(base_dir)
    if ext_root not in sys.path:
        sys.path.insert(0, ext_root)
    from indextts.infer_v2 import QwenEmotion

    qwen = QwenEmotion(model_dir)
    _QWEN_CACHE = qwen
    return qwen

class IndexTTS2EmotionFromText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
            },
        }

    RETURN_TYPES = ("EMOTION_VECTOR", "STRING")
    FUNCTION = "build"
    CATEGORY = "Audio/IndexTTS"

    def build(self, text: str) -> Tuple[list, str]:
        if not isinstance(text, str) or len(text.strip()) == 0:
            raise ValueError("Text is empty. Please provide descriptive emotion text.")

        base_dir = os.path.dirname(os.path.abspath(__file__))
        qwen_dir = os.path.join(os.path.dirname(base_dir), "checkpoints", "qwen0.6bemo4-merge")
        if not os.path.isdir(qwen_dir):
            raise FileNotFoundError(f"QwenEmotion model not found at: {qwen_dir}")

        qwen = _get_qwen(qwen_dir)
        emo_dict = qwen.inference(text)

        order = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]
        vec = [float(max(0.0, emo_dict.get(k, 0.0))) for k in order]

        #Clamp individual max to 1.4 (UI/Gradio baseline)
        vec = [min(1.4, v) for v in vec]

        cap = 1.5
        total = sum(vec)
        if total > cap:
            raise ValueError(
                f"Emotion vector sum {total:.3f} exceeds maximum {cap}. Reduce intensities or adjust with the Emotion Vector node."
            )

        info = (
            f"Detected emotion vector (sum={sum(vec):.2f}):\n"
            + ", ".join(f"{k}={v:.2f}" for k, v in zip(order, vec))
        )
        return vec, info
