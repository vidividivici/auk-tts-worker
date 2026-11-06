"""RunPod serverless worker for AuK (Tencent Hunyuan) zero-shot voice cloning. English and Chinese only.

Same contract as longcat-tts-worker: `warmup`, `generate {text, ref_url, text_url}` -> `{audio: base64 wav}`.
Voices (reference wav + transcript) live in Vercel Blob and arrive as URLs, so the worker is stateless.
"""
import base64, hashlib, io, os, sys, time, urllib.request

T0 = time.time()
import numpy as np, soundfile as sf, torch
import runpod
from auk.infer.infer_auk import AukInfer, get_gen_duration

AUK_DIR = os.environ.get("AUK_DIR", "/models/auk")
QWEN_DIR = os.environ.get("QWEN_DIR", "/models/qwen")
CPU_OFFLOAD = os.environ.get("AUK_CPU_OFFLOAD", "0") == "1"
CACHE = "/tmp/voices"
MAX_OUT_S = 45.0   # longest clip per request (the UI caps the text at 600 characters)
MAX_CHARS = 600

ckpt = next(f for f in os.listdir(AUK_DIR) if f.startswith("auk") and f.endswith(".safetensors"))
eng = AukInfer(f"{AUK_DIR}/config.yaml", f"{AUK_DIR}/{ckpt}", qwen_path=QWEN_DIR, cpu_offload=CPU_OFFLOAD)
print(f"{eng.config.model.name} ready in {time.time() - T0:.1f}s (cpu_offload={CPU_OFFLOAD})", flush=True)


def fetch(url, ext=""):
    # voice files are immutable blobs (random-suffixed names), so cache them for the worker's lifetime
    os.makedirs(CACHE, exist_ok=True)
    path = f"{CACHE}/{hashlib.sha1(url.encode()).hexdigest()}{ext}"
    if not os.path.exists(path):
        with urllib.request.urlopen(url, timeout=30) as r:
            data = r.read()
        open(path + ".part", "wb").write(data); os.replace(path + ".part", path)
    return path


def wav_b64(x, sr):
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


def generate(ref_path, prompt_text, text, max_out_s=MAX_OUT_S):
    text = (text or "").strip()
    if not text:
        return {"error": "Empty text"}
    if len(text) > MAX_CHARS:
        return {"error": f"Text too long for AuK ({len(text)} characters, max {MAX_CHARS})"}
    # AuK is non-autoregressive: the output length is set up front from the reference's speaking rate
    est = get_gen_duration(ref_path, prompt_text, text)
    if est > max_out_s * 1.1:  # squeezing more speech into the 45 s window garbles it (measured), so refuse instead
        return {"error": f"Text too long for this voice with AuK (about {est:.0f} s of speech, max {max_out_s:.0f} s). Shorten it or split it."}
    secs = min(max_out_s, max(1.0, est))
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": f'Say the following with the same voice: "{text}"'},
        {"type": "audio", "audio": ref_path},
    ]}]
    a, sr = eng.generate(msgs, gen_seconds=secs, seed=int(np.random.randint(0, 2**31)))
    return {"audio": wav_b64(np.clip(a.squeeze().numpy(), -1, 1), sr)}


def handler(job):
    inp = job["input"]
    action = inp.get("action")
    if action == "warmup":
        return {"ok": True}
    if action == "generate":
        if not (inp.get("ref_url") and inp.get("text_url")):
            return {"error": "Voice not found"}
        ref = fetch(inp["ref_url"], ".wav")
        pt = open(fetch(inp["text_url"]), encoding="utf-8").read().strip()
        return generate(ref, pt, inp.get("text"))
    return {"error": f"unknown action {action}"}


def warmup():
    # the first generation after boot is slower (CUDA kernel loading); pay that before taking jobs
    t = time.time()
    path = "/tmp/warmup.wav"
    sf.write(path, (np.random.randn(3 * 24000) * 0.01).astype(np.float32), 24000)
    generate(path, "Hello there.", "Hi.", max_out_s=1.0)
    print(f"warmup generation {time.time() - t:.1f}s, total boot {time.time() - T0:.1f}s", flush=True)


if __name__ == "__main__":
    if "--test_input" not in sys.argv:
        warmup()
    runpod.serverless.start({"handler": handler})
