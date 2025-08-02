"""
Extract question-answer pairs from PDF textbooks, write the output to a JSONL file with the format:
{"instruction": <question>, "response": <answer>}

"""

import fitz
import jsonlines
import sys
import re


EXAMPLE_CHAPTERS = ['5', '7']
EXERCISE_CHAPTERS = ['5', '7']

TOPIC_PATTERNS = [
    r"confidence intervals?",
    r"\bCI(?:s)?\b",
    r"margins? of error",
    r"confidence levels?",
    r"hypothesis tests?",
    r"hypothesis testing",
    r"\bnull hypothesis\b",
    r"\balternative hypothesis\b",
    r"significance levels?",
    r"\balpha\b",
    r"p[-\s]?values?",
    r"(?:t|z)[-\s]?tests?\b",
    r"\btest statistics?\b",
    r"critical values?",
    r"reject(?:ing)? the null",
    r"fail to reject",
    r"one[-\s]?sample",
    r"two[-\s]?sample",
    r"\bintervals?\b",
]

# Build a single regex that matches any of the above, case‑insensitive
TOPIC_RE = re.compile(r"(?i)\b(?:" + "|".join(TOPIC_PATTERNS) + r")\b")

def extract_qapairs(pdf_path, output_path):
    """
    1. Aim: Extract all QA pairs from a designated PDF and write to JSONL file 
    2. Regex used to find example, excercise, problem blocks.
    3. Split each block into question - answer pairs.
    4. Remove noise and convert to a JSONL file. 

    """

    doc = fitz.open(pdf_path)
    full_text = "\n".join(page.get_text() for page in doc)
    
    qa_list = []


    #gather in chapter examples 
    for chap in EXAMPLE_CHAPTERS:
        pattern = re.compile(
            rf"(Example {chap}\.\d+[\s\S]*?)(?="
            rf"Example {chap}\.\d+|Example [^ ]|^\d+\.|\Z)",
            re.IGNORECASE | re.MULTILINE
        )
        for block in pattern.findall(full_text):
            q, a = split_block(block)
            if q and a and topic_match(q):
                qa_list.append((q, a))

    # Find the start of Appendix A (Exercise solutions) and end before Appendix B
    appendix_a_start = re.search(r"Appendix A\s*Exercise solutions", full_text, re.IGNORECASE)
    appendix_b_start = re.search(r"Appendix B", full_text, re.IGNORECASE)
    
    # Extract main text (everything before Appendix A)
    main_text = full_text[: appendix_a_start.start()] if appendix_a_start else full_text
    
    # Extract only Appendix A (between Appendix A and Appendix B)
    if appendix_a_start and appendix_b_start:
        appendix_a = full_text[appendix_a_start.start():appendix_b_start.start()]
    elif appendix_a_start:
        appendix_a = full_text[appendix_a_start.start():]
    else:
        appendix_a = ""

    for chap in EXERCISE_CHAPTERS:
        #question pattern from chapter / main text
        q_pattern = re.compile(
            rf"^{chap}\.\d+.*?(?=^\d+\.\d+|^$)",
            re.MULTILINE | re.DOTALL
        )
        questions = q_pattern.findall(main_text)
        #solution patterns from appendix A only
        s_pattern = re.compile(
            rf"({chap}\.\d+)\s+([\s\S]*?)(?=^\d+\.\d+|^$)",
            re.MULTILINE
        )
        sol_map = {m.group(1): m.group(2) for m in s_pattern.finditer(appendix_a)}
        #map of questions to solutions 
        for q_block in questions:
            num = re.match(rf"({chap}\.\d+)", q_block).group(1)
            if num in sol_map:
                q = clean_text(q_block)
                a = clean_text(sol_map[num])
                if topic_match(q):
                    qa_list.append((q, a))


    # Write to JSONL file
    with jsonlines.open(output_path, mode='w') as writer:
        for ques, ans in qa_list:
            writer.write({"instruction": ques, "response": ans})


def split_block(block):
    """
    Split an example block into (question, answer).
    1) Try explicit 'Solution:' or 'Answer:' markers.
    2) Fallback: split at first blank line or punctuation rule line.
    """
    # 1) explicit marker
    parts = re.split(r"(?:Solution:|Answer:)", block, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) >= 2:
        q, a = parts[0], parts[1]
    else:
        # 2) fallback: blank line(s) or rule line (---/___)
        parts2 = re.split(r"\n{2,}|[-–—_]{3,}\n", block, maxsplit=1)
        if len(parts2) >= 2:
            q, a = parts2[0], parts2[1]
        else:
            return None, None

    return clean_text(q), clean_text(a)

def topic_match(text):
    """Return True if text contains any CI or hypothesis-test keyword."""
    return bool(TOPIC_RE.search(text))


def clean_text(s):
    """Collapse whitespace & strip leading/trailing spaces."""
    return re.sub(r"\s+", " ", s).strip()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: extract_qa.py <input.pdf> <output.jsonl>")
        sys.exit(1)
    extract_qapairs(sys.argv[1], sys.argv[2])
    print(f"Extracted QA to {sys.argv[2]}")