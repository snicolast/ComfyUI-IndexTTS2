from typing import Tuple


class IndexTTS2EmotionVector:
    @classmethod
    def INPUT_TYPES(cls):
        slider = {"min": 0.0, "max": 1.4, "step": 0.05}
        return {
            "required": {
                "happy": ("FLOAT", {"default": 0.0, **slider}),
                "angry": ("FLOAT", {"default": 0.0, **slider}),
                "sad": ("FLOAT", {"default": 0.0, **slider}),
                "afraid": ("FLOAT", {"default": 0.0, **slider}),
                "disgusted": ("FLOAT", {"default": 0.0, **slider}),
                "melancholic": ("FLOAT", {"default": 0.0, **slider}),
                "surprised": ("FLOAT", {"default": 0.0, **slider}),
                "calm": ("FLOAT", {"default": 0.0, **slider}),
            },
        }

    RETURN_TYPES = ("EMOTION_VECTOR",)
    FUNCTION = "build"
    CATEGORY = "Audio/IndexTTS"

    def build(self,
              happy: float,
              angry: float,
              sad: float,
              afraid: float,
              disgusted: float,
              melancholic: float,
              surprised: float,
              calm: float) -> Tuple[list]:
        vec = [
            float(max(0.0, happy)),
            float(max(0.0, angry)),
            float(max(0.0, sad)),
            float(max(0.0, afraid)),
            float(max(0.0, disgusted)),
            float(max(0.0, melancholic)),
            float(max(0.0, surprised)),
            float(max(0.0, calm)),
        ]
        total = sum(vec)
        cap = 1.5
        if total > cap:
            raise ValueError(f"Emotion vector sum {total:.3f} exceeds maximum {cap}. Lower one or more sliders so the sum ≤ {cap}.")
        return (vec,)
