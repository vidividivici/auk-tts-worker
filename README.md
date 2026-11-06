# auk-tts-worker

RunPod serverless worker for [AuK](https://github.com/Tencent-Hunyuan/AuK) (Tencent Hunyuan, MIT) zero-shot voice cloning. **English and Chinese only.**
The AuK checkpoint and the Qwen2.5-Omni-3B thinker (LLM + audio encoder only; talker, token2wav and vision weights dropped) are baked into the image.

Actions (`input.action`): `warmup`, `generate {text, ref_url, text_url}` → `{audio: base64 wav (24 kHz, 16-bit)}`.
Same contract as [longcat-tts-worker](https://github.com/vidividivici/longcat-tts-worker); voices live in Vercel Blob.
Text is limited to 600 characters and the output to 45 s per request.

Build arg `AUK_VARIANT` selects `AuK-Flash` (4-step distilled, default) or `AuK` (32-step base).
Env `AUK_CPU_OFFLOAD=1` keeps the encoder and DiT in CPU RAM between steps (less VRAM, slower).

Image: `ghcr.io/vidividivici/auk-tts-worker:latest` (built by GitHub Actions on push to main).
