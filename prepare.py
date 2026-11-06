"""Build-time: bake AuK (or AuK-Flash) and the Qwen2.5-Omni-3B *thinker* (LLM + audio encoder) into the image.
AuK only uses the Omni thinker as its text/audio encoder, so the talker, token2wav and vision weights
(~4.5 GB of the 12 GB checkpoint) are dropped, and the weights are re-sharded at 2 GB to keep build RAM low."""
import json, os, shutil, sys
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

VARIANT = sys.argv[1]  # "AuK" or "AuK-Flash"
REVS = {"AuK": "790742b71a4430120daf2b2099192abae449eb9f", "AuK-Flash": "575b92f0895f75180bf2cbd35f2e176c5732b8ed"}
QWEN_REV = "f75b40e3da2003cdd6e1829b1f420ca70797c34e"
KEEP = ("thinker.model.", "thinker.audio_tower.", "thinker.lm_head.")
SHARD = 2 * 1024**3

snapshot_download(f"tencent/{VARIANT}", revision=REVS[VARIANT], local_dir="/models/auk", allow_patterns=["config.yaml", "*.safetensors", "LICENSE"])
shutil.rmtree("/models/auk/.cache", ignore_errors=True)

src = snapshot_download("Qwen/Qwen2.5-Omni-3B", revision=QWEN_REV, cache_dir="/tmp/hf_src")
out = "/models/qwen"
os.makedirs(out, exist_ok=True)
for f in os.listdir(src):
    if not f.endswith(".safetensors") and f != "model.safetensors.index.json":
        shutil.copy(os.path.join(src, f), out)
# the vision tower is deleted right after loading (AuK never uses it); depth 0 avoids building and randomly initialising it
cfg = json.load(open(os.path.join(out, "config.json")))
cfg["thinker_config"]["vision_config"]["depth"] = 0
json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)

idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
files = sorted({f for k, f in idx.items() if k.startswith(KEEP)})
shards, cur, size = [], {}, 0
def flush():
    global cur, size
    if cur:
        shards.append({k: v.numel() * v.element_size() for k, v in cur.items()})
        save_file(cur, os.path.join(out, f"part{len(shards)}.tmp"), metadata={"format": "pt"})
        cur, size = {}, 0
for f in files:
    with safe_open(os.path.join(src, f), "pt") as sf:
        for k in sf.keys():
            if k.startswith(KEEP):
                t = sf.get_tensor(k).contiguous()
                cur[k] = t; size += t.numel() * t.element_size()
                if size >= SHARD:
                    flush()
flush()
n, wm, total = len(shards), {}, 0
for i, s in enumerate(shards, 1):
    name = f"model-{i:05d}-of-{n:05d}.safetensors"
    os.rename(os.path.join(out, f"part{i}.tmp"), os.path.join(out, name))
    for k, b in s.items():
        wm[k] = name; total += b
json.dump({"metadata": {"total_size": total}, "weight_map": wm}, open(os.path.join(out, "model.safetensors.index.json"), "w"), indent=1)
print(f"qwen thinker: {n} shards, {total / 1e9:.2f} GB")
shutil.rmtree("/tmp/hf_src", ignore_errors=True)
