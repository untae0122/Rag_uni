import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


# ----------------------------
# 입력: query 1개 + 문서 여러개
# ----------------------------
query = "how much protein should a female eat"

docs = [
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day...",
    "Definition of summit for English Language Learners: the highest point of a mountain...",
    # 여기에 문서 계속 추가
]

# e5 권장 prefix
query_text = f"query: {query}"
doc_texts = [f"passage: {d}" for d in docs]

# ----------------------------
# 모델/토크나이저 로드
# ----------------------------
model_path = "/home/work/Redteaming/data1/VIDEO_HALLUCINATION/hf_cache/hub/models--intfloat--e5-large-v2/snapshots/f169b11e22de13617baa190a028a32f3493550b6"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
model.eval()

# ----------------------------
# 임베딩 계산 함수
# ----------------------------
@torch.no_grad()
def encode(texts):
    batch = tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**batch)
    emb = average_pool(outputs.last_hidden_state, batch["attention_mask"])
    emb = F.normalize(emb, p=2, dim=1)  # cosine similarity를 위한 정규화
    return emb

# ----------------------------
# query vs docs 유사도 + Top-5
# ----------------------------
q_emb = encode([query_text])          # (1, d)
d_emb = encode(doc_texts)             # (N, d)

scores = (q_emb @ d_emb.T).squeeze(0) * 100  # (N,)

top_k = min(5, len(docs))
top_scores, top_idx = torch.topk(scores, k=top_k, largest=True)

print(f"Query: {query}\n")
for rank, (s, i) in enumerate(zip(top_scores.tolist(), top_idx.tolist()), start=1):
    print(f"[{rank}] score={s:.4f}  doc_index={i}")
    print(docs[i])
    print("-" * 80)
