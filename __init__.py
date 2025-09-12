from .nodes.indextts2_node import IndexTTS2Simple
from .nodes.indextts2_node_emovec import IndexTTS2EmotionVector
from .nodes.indextts2_node_emotext import IndexTTS2EmotionFromText

NODE_CLASS_MAPPINGS = {
    "IndexTTS2Simple": IndexTTS2Simple,
    "IndexTTS2EmotionVector": IndexTTS2EmotionVector,
    "IndexTTS2EmotionFromText": IndexTTS2EmotionFromText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IndexTTS2Simple": "IndexTTS2 Simple",
    "IndexTTS2EmotionVector": "IndexTTS2 Emotion Vector",
    "IndexTTS2EmotionFromText": "IndexTTS2 Emotion From Text",
}
