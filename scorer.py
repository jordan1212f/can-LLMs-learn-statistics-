import json, argparse, yaml
from pathlib import Path
from collections import defaultdict

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
                print(f"[WARN] Bad JSON at {path}:{ln} -> {e}")
    return rows

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))

def build_deduction_points_map(rubric):
    return dict(rubric["scoring"].get("deductions", {}))

def get_caps(rubric):
    caps = rubric["scoring"].get("caps", {}) or {}
    return float(caps.get("deduction_cap_fraction", 1.0)), int(caps.get("per_deduction_max", 10))

def allowed_for_category(code, category, allowlist):
    if not allowlist:
        return True
    cat_rules = allowlist.get(category)
    if cat_rules is None:
        return True
    allowed = cat_rules.get("allowed_deductions", [])
    return (code in allowed) if allowed else True

def clamp(x, low, high):
    return max(low, min(high, x))
#a function which clamps the score between the rubric mark which is an interval [0, 10] 

def summarise(rows):
    by_model = defaultdict(list)
    by_cat = defaultdict(list)
    for r in rows:
        sf = r.get("score_final")
        if isinstance(sf, (int, float)):
            by_model[r.get("model", r.get("model_name","unknown"))].append(sf)
            by_cat[r.get("category","unknown")].append(sf)
    def stats(vals):
        if not vals:
            return {"n":0, "mean":None}
        return {"n": len(vals), "mean": sum(vals)/len(vals)}
    return {
        "models": {m: stats(v) for m,v in by_model.items()},
        "categories": {c: stats(v) for c,v in by_cat.items()}
    }

