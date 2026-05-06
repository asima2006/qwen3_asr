import torch
import json
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from jiwer import wer as compute_wer, cer as compute_cer
from sklearn.metrics import accuracy_score
from qwen_asr import Qwen3ASRModel

# =========================
# LOAD MODEL
# =========================
model = Qwen3ASRModel.from_pretrained(
    "./qwen3-asr-finetuning-out/checkpoint-479",
    dtype=torch.bfloat16,
    device_map="cuda:0",
)

# =========================
# LOAD DATA
# =========================
eval_data = []
with open("./eval.jsonl", "r") as f:
    for line in f:
        eval_data.append(json.loads(line))

print(f"Eval samples loaded: {len(eval_data):,}")

# =========================
# HELPERS
# =========================

def clean_text(text):
    t = text.split("<asr_text>")[-1].strip().lower()
    return t if t else None   # return None if empty

def get_audio_duration(path):
    import soundfile as sf
    try:
        info = sf.info(path)
        return info.frames / info.samplerate
    except:
        return 0

def extract_scenario(text):
    if "weather"   in text: return "weather"
    elif "call"    in text: return "call"
    elif "music"   in text: return "music"
    elif "alarm"   in text: return "alarm"
    elif "timer"   in text: return "timer"
    elif "message" in text: return "message"
    elif "remind"  in text: return "reminder"
    return "other"

def slot_f1_score(ref_text, pred_text):
    r = set(ref_text.split())
    p = set(pred_text.split())
    inter     = len(r & p)
    precision = inter / len(p) if p else 0
    recall    = inter / len(r) if r else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

# =========================
# EVALUATION LOOP
# =========================
results     = []
skipped     = 0
start_total = time.time()

for sample in tqdm(eval_data, desc="Evaluating"):
    path = sample.get("audio", "")
    raw_text = sample.get("text", "")

    ref = clean_text(raw_text)
    if not ref:          # skip empty transcripts
        skipped += 1
        continue

    try:
        t0     = time.time()
        output = model.transcribe(audio=path)
        latency = (time.time() - t0) * 1000

        pred = output[0].text.strip().lower() if output else ""
        if not pred:     # skip empty predictions
            skipped += 1
            continue

        duration = get_audio_duration(path)
        rtf      = latency / (duration * 1000) if duration > 0 else 0

        results.append({
            "ref":           ref,
            "pred":          pred,
            "latency":       latency,
            "duration":      duration,
            "rtf":           rtf,
            "ref_scenario":  extract_scenario(ref),
            "pred_scenario": extract_scenario(pred),
            "slot_f1":       slot_f1_score(ref, pred),
        })

    except Exception as e:
        skipped += 1
        continue

end_total = time.time()
print(f"\nEvaluated: {len(results):,}  |  Skipped: {skipped}")

# =========================
# DATAFRAME + METRICS
# =========================
ok = pd.DataFrame(results)

refs = ok["ref"].tolist()
preds = ok["pred"].tolist()

# jiwer requires non-empty strings — guaranteed above
overall_wer = compute_wer(refs, preds)
overall_cer = compute_cer(refs, preds)

lat        = ok["latency"].values
throughput = len(ok) / (end_total - start_total)
scen_acc   = accuracy_score(ok["ref_scenario"], ok["pred_scenario"])

# =========================
# FINAL TABLE
# =========================
rows = [
    ["Metric",       "Value"],
    ["Samples",      f"{len(ok):,}"],
    ["Skipped",      f"{skipped:,}"],
    ["WER",          f"{overall_wer*100:.2f}%"],
    ["CER",          f"{overall_cer*100:.2f}%"],
    ["P50 Latency",  f"{np.percentile(lat, 50):.0f} ms"],
    ["P90 Latency",  f"{np.percentile(lat, 90):.0f} ms"],
    ["P99 Latency",  f"{np.percentile(lat, 99):.0f} ms"],
    ["Throughput",   f"{throughput:.2f} utt/s"],
    ["RTF (avg)",    f"{ok['rtf'].mean():.3f}x"],
    ["Scenario Acc", f"{scen_acc*100:.2f}%"],
    ["Slot F1",      f"{ok['slot_f1'].mean()*100:.2f}%"],
]

print(f"\n{'='*35}")
for r in rows:
    print(f"  {r[0]:<15s}: {r[1]}")
print(f"{'='*35}")

# Styled display (notebook)
display(pd.DataFrame(rows[1:], columns=rows[0]).style.set_properties(
    **{"text-align": "left", "font-size": "14px"}
))

# =========================
# SAMPLE PREDICTIONS
# =========================
print("\n--- Sample Predictions (first 10) ---")
for _, r in ok.head(10).iterrows():
    print(f"  REF  : {r['ref']}")
    print(f"  PRED : {r['pred']}")
    print(f"  WER  : {compute_wer([r['ref']], [r['pred']])*100:.1f}%  |  Latency: {r['latency']:.0f}ms")
    print()
