import torch
import json
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from jiwer import wer, cer
from sklearn.metrics import accuracy_score, f1_score
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

# =========================
# HELPERS
# =========================

def get_audio_duration(path):
    import soundfile as sf
    try:
        info = sf.info(path)
        return info.frames / info.samplerate
    except:
        return 0

def extract_scenario(text):
    # simple rule-based (customize if needed)
    text = text.lower()
    if "weather" in text:
        return "weather"
    elif "call" in text:
        return "call"
    elif "music" in text:
        return "music"
    return "other"

def extract_slots(text):
    # dummy slot extraction (customize for your task)
    words = set(text.lower().split())
    return words

# =========================
# EVALUATION LOOP
# =========================
results = []

start_total = time.time()

for sample in tqdm(eval_data, desc="Evaluating"):
    path = sample["audio"]   # ✅ FIXED
    ref  = sample["text"].strip().lower()

    try:
        t0 = time.time()
        output = model.transcribe(audio=path)
        latency = (time.time() - t0) * 1000

        pred = output[0].text.strip().lower()

        duration = get_audio_duration(path)
        rtf = latency / (duration * 1000) if duration > 0 else 0

        results.append({
            "ref": ref,
            "pred": pred,
            "latency": latency,
            "duration": duration,
            "rtf": rtf,
        })

    except Exception as e:
        continue

end_total = time.time()

# =========================
# CONVERT TO DATAFRAME
# =========================
ok = pd.DataFrame(results)

# =========================
# METRICS
# =========================

# WER / CER
overall_wer = wer(ok["ref"], ok["pred"])
overall_cer = cer(ok["ref"], ok["pred"])

# latency stats
lat = ok["latency"].values

# throughput
total_time = end_total - start_total
throughput = len(ok) / total_time

# scenario accuracy
pred_scenarios = ok["pred_scenario"]
scenario_acc = accuracy_score(ok["ref_scenario"], pred_scenarios)

# slot F1 (simple set overlap)
slot_f1_list = []
for r, p in zip(ok["ref_slots"], ok["pred_slots"]):
    if len(r) == 0 and len(p) == 0:
        slot_f1_list.append(1.0)
        continue
    inter = len(r & p)
    precision = inter / len(p) if len(p) > 0 else 0
    recall = inter / len(r) if len(r) > 0 else 0
    if precision + recall == 0:
        slot_f1_list.append(0)
    else:
        slot_f1_list.append(2 * precision * recall / (precision + recall))

ok["slot_f1"] = slot_f1_list

# =========================
# FINAL TABLE
# =========================

rows = [
    ["Metric",       "Value"],
    ["Samples",      f"{len(ok):,}"],
    ["WER",          f"{overall_wer*100:.2f}%"],
    ["CER",          f"{overall_cer*100:.2f}%"],
    ["P50 Latency",  f"{np.percentile(lat, 50):.0f} ms"],
    ["P90 Latency",  f"{np.percentile(lat, 90):.0f} ms"],
    ["P99 Latency",  f"{np.percentile(lat, 99):.0f} ms"],
    ["Throughput",   f"{throughput:.2f} utt/s"],
    ["RTF (avg)",    f"{ok['rtf'].mean():.3f}x"],
    ["Scenario Acc", f"{scenario_acc*100:.2f}%"],
    ["Slot F1",      f"{ok['slot_f1'].mean()*100:.2f}%"],
]

# =========================
# PRINT TABLE
# =========================

for r in rows:
    print(f"{r[0]:<15} : {r[1]}")