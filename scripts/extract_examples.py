"""
Extract example question-answer pairs from PDF textbooks, write the output to a JSONL file.
Examples are self-contained blocks that include both question and answer.
"""

import fitz
import jsonlines
import sys
import re

EXAMPLE_CHAPTERS = ['5', '7']

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

def extract_examples(pdf_path, output_path):
    """
    Extract self-contained example Q&A pairs from PDF chapters.
    """
    doc = fitz.open(pdf_path)
    full_text = "\n".join(page.get_text() for page in doc)
    
    qa_list = []

    # Extract examples from specified chapters
    for chap in EXAMPLE_CHAPTERS:
        # Find all examples in the chapter - working pattern
        pattern = re.compile(
            rf"(Example {chap}\.\d+.*?)(?=Example|GUIDED PRACTICE|^\d+\.\d+|Exercises)",
            re.IGNORECASE | re.MULTILINE | re.DOTALL
        )
        for block in pattern.findall(full_text):
            q, a = split_block(block)
            if q and a:  # Remove topic filtering for now
                qa_list.append((q, a))

    # Write to JSONL file
    with jsonlines.open(output_path, mode='w') as writer:
        for ques, ans in qa_list:
            writer.write({"instruction": ques, "response": ans})

def split_block(block):
    """
    Split an example block into (question, answer).
    1) Try explicit 'Solution:' or 'Answer:' markers.
    2) Fallback: split after the first paragraph (question) - everything else is the answer.
    """
    # 1) explicit marker
    parts = re.split(r"(?:Solution:|Answer:)", block, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) >= 2:
        q, a = parts[0], parts[1]
    else:
        # 2) For examples without explicit markers, split after first paragraph
        # Look for the pattern: EXAMPLE X.Y followed by question, then answer
        example_match = re.match(r"(EXAMPLE\s+\d+\.\d+\s*\n*.*?)(\n.*)", block, re.DOTALL | re.IGNORECASE)
        if example_match:
            header_and_question = example_match.group(1)
            answer = example_match.group(2)
            
            # Split the header and question part at the first line break after substantial text
            lines = header_and_question.split('\n')
            if len(lines) >= 2:
                # Take the header and first substantial line(s) as question
                question_lines = []
                answer_lines = []
                found_question = False
                
                for line in lines:
                    if line.strip().startswith('EXAMPLE'):
                        question_lines.append(line)
                    elif not found_question and line.strip():
                        question_lines.append(line)
                        found_question = True
                    elif found_question and line.strip():
                        answer_lines.append(line)
                
                if question_lines and (answer_lines or answer.strip()):
                    q = '\n'.join(question_lines)
                    a = '\n'.join(answer_lines) + answer if answer_lines else answer
                else:
                    return None, None
            else:
                return None, None
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
        print("Usage: extract_examples.py <input.pdf> <output.jsonl>")
        sys.exit(1)
    extract_examples(sys.argv[1], sys.argv[2])
    print(f"Extracted Examples to {sys.argv[2]}")
