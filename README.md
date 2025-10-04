ComfyUI-IndexTTS2
=================

Lightweight ComfyUI wrapper for IndexTTS 2 (voice cloning + emotion control). Nodes call the upstream inference code so behaviour stays matched with the original repo.

Original repo: https://github.com/index-tts/index-tts

## Updates
- 2025-09-22: Added IndexTTS2 Advanced node exposing sampling, speed, seed, and other generation controls.

## Install
- Clone this repository into `ComfyUI/custom_nodes/`
- Inside your ComfyUI Python environment:
  ```bash
  pip install wetext
  pip install -r requirements.txt
  ```

## Models
- Create `checkpoints/` in the repo root and copy the IndexTTS-2 release there (https://huggingface.co/IndexTeam/IndexTTS-2/tree/main). Missing files will be cached from Hugging Face automatically, but a full local copy keeps everything offline.
- For full offline use download once and place the files below:
  - `facebook/w2v-bert-2.0` -> `checkpoints/w2v-bert-2.0/` (the loader checks this folder before contacting Hugging Face)
  - BigVGAN config and weights -> `checkpoints/bigvgan/`
  - MaskGCT semantic codec -> `checkpoints/semantic_codec/model.safetensors`
  - CAMPPlus model -> `checkpoints/campplus_cn_common.bin`
  - Optional: QwenEmotion (`qwen0.6bemo4-merge/`) for the text-to-emotion helper node
- Typical layout:
  ```
  checkpoints/
    config.yaml, gpt.pth, s2mel.pth, bpe.model, feat*.pt, wav2vec2bert_stats.pt
    bigvgan/{config.json,bigvgan_generator.pt}
    semantic_codec/model.safetensors
    campplus_cn_common.bin
    qwen0.6bemo4-merge/[model files]
    w2v-bert-2.0/[HF files]
  ```

## Nodes
- **IndexTTS2 Simple** – speaker audio, text, optional emotion audio/vector; outputs audio + status string. Auto-selects device, FP16 on CUDA.
- **IndexTTS2 Advanced** – Simple inputs plus overrides for sampling, speech speed, pauses, CFG, seed.
- **IndexTTS2 Emotion Vector** – eight sliders (0.0–1.4, sum <= 1.5) producing an emotion vector.
- **IndexTTS2 Emotion From Text** – requires ModelScope and local QwenEmotion; turns short text into an emotion vector + summary.

## Examples
- Speaker audio -> IndexTTS2 Simple -> Preview/Save Audio
- Speaker + emotion audio -> IndexTTS2 Simple -> Save
- Emotion Vector -> IndexTTS2 Simple -> Save
- Emotion From Text -> IndexTTS2 Simple -> Save

![ComfyUI-IndexTTS2 nodes](images/overview.png)

## Troubleshooting
- Windows only so far; DeepSpeed is disabled.
- Install `wetext` if the module is missing on first launch.
- If w2v-bert keeps downloading, confirm `checkpoints/w2v-bert-2.0/` exists (or set `W2V_BERT_LOCAL_DIR`).
- 404 or load failures usually mean a missing file in `checkpoints/`; re-check the tree above.
- Emotion vector sum must stay <= 1.5.
- BigVGAN CUDA kernel warnings are expected; PyTorch fallback kicks in automatically.

## Logs you should see
- `Loading config.json from local directory`
- `SeamlessM4TFeatureExtractor loaded from: checkpoints/w2v-bert-2.0/`
- Model paths pointing at your `checkpoints/` tree.





