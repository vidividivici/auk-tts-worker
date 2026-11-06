FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 HF_HOME=/models/hf HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y --no-install-recommends git libsndfile1 ffmpeg && rm -rf /var/lib/apt/lists/*
# AuK inference dependencies only (no Gradio / Prompt Enhancer / cloud ASR extras)
RUN pip install --no-cache-dir transformers==4.57.0 qwen-omni-utils omegaconf torchdiffeq x_transformers safetensors accelerate \
    soundfile librosa runpod huggingface_hub hf_transfer
RUN pip install --no-cache-dir --no-deps "git+https://github.com/Tencent-Hunyuan/AuK@0d40366b1039e37c6cad008ab4713f6ad6d954b7" \
    && python -c "import torchaudio, auk.infer.infer_auk"

# AUK_VARIANT: AuK-Flash (4-step distilled, CFG off) or AuK (32-step base)
ARG AUK_VARIANT=AuK-Flash
COPY prepare.py /app/prepare.py
RUN python /app/prepare.py ${AUK_VARIANT}

COPY handler.py /app/handler.py
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CMD ["python", "-u", "/app/handler.py"]
