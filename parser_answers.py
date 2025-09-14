import json, re
"""Answer files contain a part A and part B due to the prompt used to create those answers
Keeping part B confused the judge and only part A is needed for the judge.."""
def split_part_a_b(answer_text: str):
    searcher = re.search(r'\bPart\s*B\)\s*(\{.*)$', answer_text, flags=re.S)
    if not searcher:
        return answer_text.strip(), None
    part_a = answer_text[:searcher.start()].strip()
    tail = searcher.group(1).strip()
    try:
        jmatch = re.search(r'\{.*\}', tail, flags=re.S)
        ans_json = json.loads(jmatch.group(0)) if jmatch else None
    except Exception:
        ans_json = None
    return part_a, ans_json

def normalise_model_file(in_path, out_min_path, out_struct_path=None):
    fout_struct = open(out_struct_path, "w", encoding="utf-8") if out_struct_path else None
    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_min_path, "w", encoding="utf-8") as fout_min:
        for line in fin:
            if not line.strip(): 
                continue
            obj = json.loads(line)
            qid = obj.get("question_id") or obj.get("id")
            atext = obj.get("answer_text","")
            part_a, part_b_json = split_part_a_b(atext)
            # minimal line for the judge
            fout_min.write(json.dumps({"question_id": qid, "answer_text": part_a}, ensure_ascii=False) + "\n")
            # optional structured dump
            if fout_struct and part_b_json is not None:
                fout_struct.write(json.dumps({"question_id": qid, "answer_json": part_b_json}, ensure_ascii=False) + "\n")
    if fout_struct:
        fout_struct.close()


normalise_model_file("ci_rag_20250818-034631.jsonl", "ci_answers.min.jsonl", "ci_answers.structured.jsonl")
normalise_model_file("ht_rag_20250818-034631.jsonl", "ht_answers.min.jsonl", "ht_answers.structured.jsonl")
normalise_model_file("base_no_rag_20250818-034631.jsonl", "base_answers.min.jsonl", "base_answers.structured.jsonl")
