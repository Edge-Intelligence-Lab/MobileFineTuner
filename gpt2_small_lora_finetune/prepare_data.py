import os, json, glob, random, csv
from typing import List, Dict
from transformers import GPT2TokenizerFast

def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

BASE = repo_root()
MMLU_DATA_DIR = os.path.join(BASE, "data/mmlu/data")
OUT_DIR = os.path.join(BASE, "runs/mmlu_jsonl_gpt2_s128")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "pretrained")
SEQ_LEN = 128
SEED = 123
SPLIT_RATIO = 0.9  # 90% train / 10% valid

random.seed(SEED)

tok = GPT2TokenizerFast.from_pretrained(MODEL_DIR)
EOS = tok.eos_token_id or 50256

os.makedirs(OUT_DIR, exist_ok=True)

def parse_csv(path: str) -> List[Dict]:
    items: List[Dict] = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 6:
                continue
            q = row[0]
            a, b, c, d = row[1], row[2], row[3], row[4]
            ans = row[5].strip()
            if ans and ans[0] in 'ABCDabcd':
                ans = ans[0].upper()
            else:
                continue
            items.append({'question': q, 'A': a, 'B': b, 'C': c, 'D': d, 'answer': ans})
    return items

def build_ids_mask(ex):
    prompt = (
        f"Question: {ex['question']}\n"
        f"A. {ex['A']}\n"
        f"B. {ex['B']}\n"
        f"C. {ex['C']}\n"
        f"D. {ex['D']}\n"
        f"Answer: "
    )
    answer = ex['answer']
    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    ans_ids = tok.encode(" " + answer, add_special_tokens=False)
    ids = prompt_ids + ans_ids + [EOS]
    if len(ids) > SEQ_LEN:
        return None
    mask = [0]*len(prompt_ids) + [1]*len(ans_ids) + [0]
    if len(ids) < SEQ_LEN:
        pad = SEQ_LEN - len(ids)
        ids = ids + [EOS]*pad
        mask = mask + [0]*pad
    return ids, mask

all_items = []
for csv_path in glob.glob(os.path.join(MMLU_DATA_DIR, '**', '*.csv'), recursive=True):
    all_items.extend(parse_csv(csv_path))

random.shuffle(all_items)

pairs = []
for ex in all_items:
    r = build_ids_mask(ex)
    if r is None:
        continue
    pairs.append(r)

n_train = int(len(pairs) * SPLIT_RATIO)
train_pairs = pairs[:n_train]
valid_pairs = pairs[n_train:]

with open(os.path.join(OUT_DIR, 'train.jsonl'), 'w', encoding='utf-8') as f:
    for ids, mask in train_pairs:
        f.write(json.dumps({'ids': ids, 'mask': mask}, ensure_ascii=False) + '\n')
with open(os.path.join(OUT_DIR, 'valid.jsonl'), 'w', encoding='utf-8') as f:
    for ids, mask in valid_pairs:
        f.write(json.dumps({'ids': ids, 'mask': mask}, ensure_ascii=False) + '\n')

print(f"Done. train={len(train_pairs)} valid={len(valid_pairs)} out={OUT_DIR}")


