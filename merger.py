import json
import argparse
from pathlib import Path

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARNNING!!!!!] Skipping bad JSON at {path}:{ln} -> {e}")
    return rows

def stringify_gold_answer(ans):
    if ans is None:
        return ""
    if isinstance(ans, str):
        return ans
    return json.dumps(ans, ensure_ascii=False)

def build_qindex(question_rows):
    idx = {}
    for r in question_rows:
        qid = r.get("question_id") or r.get("id")
        if not qid:
            continue
        idx[qid] = {
            "question_text": r.get("question", ""),
            "gold_answer": stringify_gold_answer(r.get("answer")), 
            "category": r.get("category", "")
        }
    return idx

def merge_files(
    questions_path,
    model_min_path,
    out_path,
    model_name
):
    questions = load_jsonl(questions_path)
    qindex = build_qindex(questions)

    model_rows = load_jsonl(model_min_path)

    matched, miss_q, miss_m = 0, 0, 0
    with open(out_path, "w", encoding="utf-8") as out:
        for r in model_rows:
            qid = r.get("question_id") or r.get("id")
            if not qid:
                miss_m += 1
                continue
            qinfo = qindex.get(qid)
            if not qinfo:
                miss_q += 1
                continue

            merged = {
                "question_id": qid,
                "model_name": model_name,
                "category": qinfo.get("category", ""),
                "question_text": qinfo.get("question_text", ""),
                "gold_answer": qinfo.get("gold_answer", ""),    
                "model_answer_text": r.get("answer_text", "")
}
            out.write(json.dumps(merged, ensure_ascii=False) + "\n")
            matched += 1

    print(f"[MERGE] wrote: {matched} rows -> {out_path}")
    if miss_q:
        print(f"[WARN] {miss_q} model answers had question_ids not found in questions file.")
    if miss_m:
        print(f"[WARN] {miss_m} model rows missing question_id.")

def main():
    ap = argparse.ArgumentParser(description="Merged questions + gold answers with model answers for judging.")
    ap.add_argument("--questions", required=True, help="statistics_eval_questions.jsonl")
    ap.add_argument("--model-min", required=True, help="model answers min jsonl (question_id, answer_text)")
    ap.add_argument("--model-name", required=True, help="Name label for this model (e.g., base, ci_rag)")
    ap.add_argument("--out", required=True, help="Output merged jsonl for judge")
    args = ap.parse_args()

    for p in [args.questions, args.model_min]:
        if not Path(p).exists():
            raise SystemExit(f"File not found: {p}")

    merge_files(args.questions, args.model_min, args.out, args.model_name)

if __name__ == "__main__":
    main()
