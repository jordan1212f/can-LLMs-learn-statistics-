import json, argparse, yaml, re
from pathlib import Path
from ollama import Client

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
                print(f"[WARNING] Skipping bad JSON at {path}:{ln} -> {e}")
    return rows
#opens a my jsonl as a list of dicts

def load_text(path):
    return Path(path).read_text(encoding="utf-8")

def load_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
#opens my marking ruberic

def render_prompt(template, context):
    def repl(m):
        key = m.group(1).strip()
        val = context.get(key, "")
        if isinstance(val, (dict, list)):
            return json.dumps(val, ensure_ascii=False)
        return str(val)
    return re.sub(r"\{\{([^}]+)\}\}", repl, template)

#purpose is to avoid building the prompt each time manually, instead use a template to build the judge prompt each time

def rubric_category(rubric, category):
    cats = rubric.get("Categories", {}) or rubric.get("categories", {})
    aliases = rubric.get("aliases", {})
    if category not in cats:
        norm = aliases.get(category, None)
        if norm and norm in cats:
            category = norm
        else:
            return None, category
    return yaml.safe_dump({category: cats[category]}, allow_unicode=True, sort_keys=False), category

def pick_rubric_snippets(rubric, category):
    scoring = rubric.get("scoring", {}) or {}

    weights_yaml = yaml.safe_dump(
        {"weights": scoring.get("weights", {})},
        allow_unicode=True, sort_keys=False
    )
    deductions_yaml = yaml.safe_dump(
        {"deductions": scoring.get("deductions", {})},
        allow_unicode=True, sort_keys=False
    )
    caps_yaml = yaml.safe_dump(
        {"caps": scoring.get("caps", {})},
        allow_unicode=True, sort_keys=False
    )

    applicability_yaml = yaml.safe_dump(
        {"deduction_applicability_codes": rubric.get("deduction_applicability_codes", {})},
        allow_unicode=True, sort_keys=False
    )
    definitions_yaml = yaml.safe_dump(
        {"deduction_definitions": rubric.get("deduction_definitions", {})},
        allow_unicode=True, sort_keys=False
    )

    cat_yaml, resolved = rubric_category(rubric, category)
    aliases_yaml = yaml.safe_dump(
        {"aliases": rubric.get("aliases", {})},
        allow_unicode=True, sort_keys=False
    )

    return (weights_yaml, deductions_yaml, caps_yaml,
            applicability_yaml, definitions_yaml,
            cat_yaml, resolved, aliases_yaml)

def get_deduction_codes(rubric):
    scoring = rubric.get("scoring", {}) or {}
    codes = scoring.get("deductions") or scoring.get("deduction_codes") or {}
    codes_yaml = yaml.safe_dump({"deductions": codes},
                                allow_unicode=True, sort_keys=False)
    return codes, codes_yaml

def get_deduction_caps_yaml(rubric):
    scoring = rubric.get("scoring", {}) or {}
    caps = scoring.get("caps", {}) or {}
    return yaml.safe_dump({"caps": caps}, allow_unicode=True, sort_keys=False)

def get_applicable_deductions_yaml(rubric, resolved_category, all_codes):
    appmap = (
        rubric.get("deduction_applicability_codes")
        or rubric.get("scoring", {}).get("deduction_applicability_codes")
        or {}
    )
    if not appmap:
        allowed = sorted(all_codes.keys())
    else:
        allowed = []
        for code, cats in appmap.items():
            if not cats:
                allowed.append(code)
            else:
                if resolved_category in cats:  # exact match
                    allowed.append(code)
        allowed = [c for c in allowed if c in all_codes]

    return yaml.safe_dump({"applicable_deductions": allowed},
                          allow_unicode=True, sort_keys=False)

import json, re

def normalize_to_obj(content):
    """Return a Python dict from Ollama content which may be dict | str | None."""
    if content is None:
        return None
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        # First try direct JSON
        try:
            return json.loads(content)
        except Exception:
            # Fall back to sanitizing a string that looks like JSON but isn't strict
            return safe_json_parse(content)
    # Any other type is unusable
    return None

def extract_first_json_object(s: str) -> str:
    s = re.sub(r'^\s*```(?:json)?\s*|\s*```\s*$', '', s, flags=re.MULTILINE)
    m = re.search(r'\{(?:[^{}]|(?R))*\}', s, flags=re.S)  # first {...}
    return m.group(0) if m else s

def normalize_common_issues(s: str) -> str:
    s = s.replace('points:null,', '"points": null,').replace('points:null', '"points": null')
    s = re.sub(r',\s*([}\]])', r'\1', s)  # trailing commas
    return s

def safe_json_parse(raw):
    if raw is None:
        raise ValueError("content is None")
    if not isinstance(raw, str):
        raw = json.dumps(raw)
    raw = extract_first_json_object(raw.strip())
    raw = normalize_common_issues(raw)
    return json.loads(raw)

#will allocate the correct ruberic items to the cateogry currently being marked 

def call_ollama_judge(prompt, model_name):
    client = Client()  #http://localhost:11434
    base = {
        "role": "user",
        "content": prompt + (
            "\n\nReturn exactly ONE JSON object with the required keys. "
            "No markdown, no code fences, no extra text."
        )
    }
    
    res = client.chat(model=model_name, messages=[base],
                      options={"temperature": 0}, format="json")
    content = res.get("message", {}).get("content")

    obj = normalize_to_obj(content)
    if obj is None:
        res2 = client.chat(model=model_name, messages=[base],
                           options={"temperature": 0})
        obj = normalize_to_obj(res2.get("message", {}).get("content"))

    if obj is None:
        return {"error": "judge_failed: empty or unusable content", "raw": repr(content)}

    return obj

def parse_json(text):
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        raise

""" MAIN RUNNER FUNCTIONS"""

def main():
    ap = argparse.ArgumentParser(description="Local judge runner (Qwen via Ollama)")
    ap.add_argument("--in", dest="in_path", required=True, help="merged model file (jsonl)")
    ap.add_argument("--rubric", required=True, help="rubric.yaml")
    ap.add_argument("--prompt", required=True, help="judge_prompt.txt")
    ap.add_argument("--model", required=True, help="qwen2.5:7b-instruct")
    ap.add_argument("--out", required=True, help="output scores jsonl")
    ap.add_argument("--max", type=int, default=0, help="max rows to grade (0=all)")
    ap.add_argument("--verbose", action="store_true", help="print per-row progress")
    args = ap.parse_args()

    rows = load_jsonl(args.in_path)
    if args.max > 0:
        rows = rows[:args.max]

    rubric = load_yaml(args.rubric)
    prompt_tmpl = load_text(args.prompt)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    print(f"[starting] grading {len(rows)} rows with model={args.model}  input={args.in_path}  out={args.out}",
          flush=True)

    written = 0
    with open(outp, "w", encoding="utf-8") as fout:
        for i, r in enumerate(rows, 1):
            qid = r["question_id"]
            cat = r.get("category", "")

            #builds ALL THE rubric YAML snippets for THIS category ONLY
            (w_yaml, d_yaml, c_yaml, a_yaml_applic, d_yaml_defs,
            cat_yaml, resolved_cat, aliases_yaml) = pick_rubric_snippets(rubric, cat)

            resolved_for_ctx = resolved_cat if cat_yaml else cat
            all_codes = (rubric.get("scoring", {}) or {}).get("deductions", {}) or {}
            applicable_yaml = get_applicable_deductions_yaml(
                rubric, resolved_for_ctx, all_codes
            )

            if not cat_yaml:
                context = {
                    "question_id": qid,
                    "model_name": r.get("model_name", ""),
                    "category": cat,
                    "resolved_category": resolved_cat if cat_yaml else cat,
                    "question_text": r.get("question_text", ""),
                    "gold_answer_optional": r.get("gold_answer", ""),
                    "model_answer_text": r.get("model_answer_text", ""),

                    "rubric_weights_yaml": w_yaml,
                    "rubric_deductions_yaml": d_yaml,
                    "rubric_caps_yaml": c_yaml,
                    "rubric_category_yaml": cat_yaml or "",   
                    "rubric_aliases_yaml": aliases_yaml,

                    "deduction_applicability_codes_yaml": a_yaml_applic,
                    "deduction_definitions_yaml": d_yaml_defs,
                    "applicable_deductions_yaml": applicable_yaml,

                    "weights.correctness": rubric["scoring"]["weights"]["correctness"],
                    "weights.completeness": rubric["scoring"]["weights"]["completeness"],
                    "weights.clarity": rubric["scoring"]["weights"]["clarity"],
                }
            else:
                context = {
                    "question_id": qid,
                    "model_name": r.get("model_name", ""),
                    "category": cat,
                    "resolved_category": resolved_cat if cat_yaml else cat,
                    "question_text": r.get("question_text", ""),
                    "gold_answer_optional": r.get("gold_answer", ""),
                    "model_answer_text": r.get("model_answer_text", ""),

                    "rubric_weights_yaml": w_yaml,
                    "rubric_deductions_yaml": d_yaml,
                    "rubric_caps_yaml": c_yaml,
                    "rubric_category_yaml": cat_yaml or "",    
                    "rubric_aliases_yaml": aliases_yaml,

                    "deduction_applicability_codes_yaml": a_yaml_applic,
                    "deduction_definitions_yaml": d_yaml_defs,
                    "applicable_deductions_yaml": applicable_yaml,

                    "weights.correctness": rubric["scoring"]["weights"]["correctness"],
                    "weights.completeness": rubric["scoring"]["weights"]["completeness"],
                    "weights.clarity": rubric["scoring"]["weights"]["clarity"],
                }
            prompt = render_prompt(prompt_tmpl, context)

            try:
                result = call_ollama_judge(prompt, args.model)
                if isinstance(result, dict) and "error" in result:
                    obj = {
                        "question_id": qid,
                        "model": r.get("model_name", ""),
                        "category": cat,
                        "error": result["error"],
                        "subscores": None,
                        "score_raw": None,
                        "deductions_applied": [],
                        "score_final": None,
                        "rationale": None
        }
                else:
                    obj = result
            except Exception as e:
                obj = {
                "question_id": qid,
                "model": r.get("model_name", ""),
                "category": cat,
                "error": f"judge_failed: {type(e).__name__}: {e}",
                "subscores": None,
                "score_raw": None,
                "deductions_applied": [],
                "score_final": None,
                "rationale": None
            }
    
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            written += 1

            if args.verbose or len(rows) < 10 or i % 10 == 0:
                print(f"[judge] graded {i}/{len(rows)}", flush=True)

    print(f"[DONE] wrote {written} rows to {outp}")

if __name__ == "__main__":
    main()