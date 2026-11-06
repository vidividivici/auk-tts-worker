# auk-tts-worker

RunPod serverless worker for [AuK](https://github.com/Tencent-Hunyuan/AuK) (Tencent Hunyuan, MIT) zero-shot voice cloning. **English and Chinese only.**
The AuK checkpoint and the Qwen2.5-Omni-3B thinker (LLM + audio encoder only; talker, token2wav and vision weights dropped) are baked into the image.

Actions (`input.action`): `warmup`, `generate {text, ref_url, text_url}` → `{audio: base64 wav (24 kHz, 16-bit)}`.
Same contract as [longcat-tts-worker](https://github.com/vidividivici/longcat-tts-worker); voices live in Vercel Blob.
Text is limited to 600 characters and the output to 45 s per request; texts whose estimated speech (from the voice's speaking rate)
is longer than ~50 s are refused, because squeezing them into 45 s garbles the audio.

Build arg `AUK_VARIANT` selects `AuK-Flash` (4-step distilled, default) or `AuK` (32-step base).
Env `AUK_CPU_OFFLOAD` (default `1` in the image): the text encoder and the DiT take turns on the GPU.

Measured on an RTX A5000 (AuK-Flash, 30 s reference + 45 s output, the worst case the worker accepts):

| mode | peak VRAM (allocated / reserved) | 10 s clip | 45 s clip | GPU pool |
|---|---|---|---|---|
| `AUK_CPU_OFFLOAD=1` | 12.0 / 12.5 GiB | 8.3-9.2 s | 11.2 s | 16 GB ($0.58/h flex) |
| `AUK_CPU_OFFLOAD=0` | 19.6 / 21.8 GiB | 1.9-2.1 s | 7.2 s | 24 GB, A5000/3090 only ($0.75/h flex) |

The base `AuK` (32 steps, CFG) needs 21.8 / 23.2 GiB and takes 13 s for a 10 s clip, so Flash is the default.

Image: `ghcr.io/vidividivici/auk-tts-worker:latest` (built by GitHub Actions on push to main).
