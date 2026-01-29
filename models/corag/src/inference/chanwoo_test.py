import json
import re
import math
from collections import Counter, defaultdict

path1 = "/home/work/Redteaming/rag-exp/corag/tmp/test_run/preds_greedy_hotpotqa_validation.jsonl"
path2 = "/home/work/Redteaming/rag-exp/results/trajectory_results/corag/generator_trajectory_shard_chanwoo.json"

# ----------------- preprocessing -----------------
def tokenize(text: str):
    text = text.lower().strip()
    # 구두점/특수문자 제거 후 공백 정리
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()

def ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def modified_precision(candidate_tokens, reference_tokens_list, n):
    cand_ngrams = Counter(ngrams(candidate_tokens, n))
    if not cand_ngrams:
        return 0.0

    max_ref_counts = Counter()
    for ref_tokens in reference_tokens_list:
        ref_counts = Counter(ngrams(ref_tokens, n))
        for ng, c in ref_counts.items():
            if c > max_ref_counts[ng]:
                max_ref_counts[ng] = c

    clipped = {ng: min(count, max_ref_counts.get(ng, 0)) for ng, count in cand_ngrams.items()}
    return sum(clipped.values()) / sum(cand_ngrams.values())

def brevity_penalty(c_len, r_len):
    if c_len == 0:
        return 0.0
    if c_len > r_len:
        return 1.0
    return math.exp(1 - (r_len / c_len))

def closest_reference_length(candidate_tokens, reference_tokens_list):
    c_len = len(candidate_tokens)
    ref_lens = [len(r) for r in reference_tokens_list]
    # BLEU 표준: 길이 차이가 가장 작은 ref, 동률이면 더 짧은 ref
    closest = min(ref_lens, key=lambda rl: (abs(rl - c_len), rl))
    return closest

def bleu_score(candidate_tokens, reference_tokens_list, max_n=4, smooth=True):
    """
    BLEU-N (default 4-gram) with optional smoothing.
    Smoothing: add-epsilon 방식(간단/안정적)
    """
    if len(candidate_tokens) == 0:
        return 0.0

    precisions = []
    eps = 1e-9
    for n in range(1, max_n+1):
        p = modified_precision(candidate_tokens, reference_tokens_list, n)
        if smooth:
            p = max(p, eps)  # 0 방지
        precisions.append(p)

    log_p = sum((1/max_n) * math.log(p) for p in precisions)
    r_len = closest_reference_length(candidate_tokens, reference_tokens_list)
    bp = brevity_penalty(len(candidate_tokens), r_len)
    return bp * math.exp(log_p)

# ----------------- load data -----------------
path1_data = {}
with open(path1, "r") as f:
    for line in f:
        obj = json.loads(line)
        path1_data[obj["id"]] = [tokenize(q) for q in obj.get("subqueries", [])]

with open(path2, "r") as f:
    raw = json.load(f)

path2_data = {}
for obj in raw:
    path2_data[obj["id"]] = [tokenize(step["subquery"]) for step in obj.get("steps", [])]

# ----------------- evaluation -----------------
def evaluate(threshold=0.30):
    """
    threshold: BLEU가 이 값 이상이면 '매칭'으로 카운트
    """
    total_steps = 0
    matched_steps = 0
    bleu_values = []

    per_id = {}

    for qid, refs in path1_data.items():
        if qid not in path2_data:
            continue
        if not refs:
            continue

        cands = path2_data[qid]
        if not cands:
            continue

        # path2의 각 step subquery를 path1 refs에 대해 best BLEU로 매칭
        step_bleus = []
        for cand in cands:
            b = bleu_score(cand, refs, max_n=4, smooth=True)
            step_bleus.append(b)

        per_id[qid] = {
            "num_refs(path1_subqueries)": len(refs),
            "num_steps(path2_subqueries)": len(cands),
            "step_bleu": step_bleus,
            "mean_step_bleu": sum(step_bleus)/len(step_bleus),
            "matched_steps(>=thr)": sum(1 for b in step_bleus if b >= threshold),
        }

        total_steps += len(step_bleus)
        matched_steps += per_id[qid]["matched_steps(>=thr)"]
        bleu_values.extend(step_bleus)

    mean_bleu = sum(bleu_values)/len(bleu_values) if bleu_values else 0.0
    match_rate = matched_steps/total_steps if total_steps else 0.0

    print("===== BLEU Matching Evaluation =====")
    print(f"Total steps (path2)          : {total_steps}")
    print(f"Matched steps (BLEU >= {threshold}) : {matched_steps}")
    print(f"Match rate                   : {match_rate:.4f}")
    print(f"Mean step BLEU               : {mean_bleu:.4f}")

    return per_id

per_id_stats = evaluate(threshold=0.8)

# 특정 id 확인 예시:
# qid = "5adbf0a255429947ff17385a"
# print(per_id_stats[qid])
