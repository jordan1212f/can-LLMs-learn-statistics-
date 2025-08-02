"""
Extract exercise question-answer pairs from PDF textbooks, write the output to a JSONL file.
Exercises are questions from chapters with solutions in Appendix A.
"""

import fitz
import jsonlines
import sys
import re

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

def extract_exercises(pdf_path, output_path):
    """
    Extract exercise questions from chapters and match with solutions from Appendix A.
    """
    doc = fitz.open(pdf_path)
    full_text = "\n".join(page.get_text() for page in doc)
    
    qa_list = []

    # Find the start of Appendix A (Exercise solutions) and end before Appendix B
    # Use exact positions found from debugging
    appendix_a_start_pos = 915133  # Known position from debugging
    appendix_b_start_pos = 1004529  # Known position from debugging
    
    # Extract main text (everything before Appendix A)
    main_text = full_text[:appendix_a_start_pos]
    
    # Extract only Appendix A (between Appendix A and Appendix B)
    appendix_a = full_text[appendix_a_start_pos:appendix_b_start_pos]
    
    print(f"Appendix A length: {len(appendix_a)}")
    print(f"Main text length: {len(main_text)}")

    # Find exercises under "Chapter Exercises" sections in each chapter
    for chap in EXERCISE_CHAPTERS:
        print(f"\nProcessing Chapter {chap}...")
        
        # Use known positions for chapters
        if chap == '5':
            chapter_5_start = 392317
            chapter_6_start = main_text.find('Chapter 6', chapter_5_start)
            if chapter_6_start == -1:
                chapter_content = main_text[chapter_5_start:]
            else:
                chapter_content = main_text[chapter_5_start:chapter_6_start]
        elif chap == '7':
            chapter_7_start = 603529
            chapter_8_start = main_text.find('Chapter 8', chapter_7_start)
            if chapter_8_start == -1:
                chapter_content = main_text[chapter_7_start:]
            else:
                chapter_content = main_text[chapter_7_start:chapter_8_start]
        else:
            continue
            
        print(f"Chapter {chap} content length: {len(chapter_content)}")
        
        # Find exercises section in this chapter
        exercises_pattern = re.compile(r"Exercises.*?$", re.DOTALL)
        exercises_match = exercises_pattern.search(chapter_content)
        
        if not exercises_match:
            print(f"No exercises section found in Chapter {chap}")
            continue
            
        exercises_content = exercises_match.group()
        print(f"Exercises section length: {len(exercises_content)}")
        
        # Find all exercise questions that start with chapter number
        # Use a more careful pattern that captures the complete question including all parts
        q_pattern = re.compile(
            rf"({chap}\.\d+)\s+([A-Z].*?)(?=\n{chap}\.\d+\s+[A-Z]|$)",
            re.MULTILINE | re.DOTALL
        )
        question_matches = list(q_pattern.finditer(exercises_content))
        
        print(f"Found {len(question_matches)} question matches")
        
        questions = []
        for match in question_matches:
            q_num = match.group(1)
            q_text = match.group(2).strip()
            
            # Only include if it looks like a real question
            # Real questions should have substantial text and start with a capital letter
            # Also check that the question number is reasonable (not like 7.5394)
            if (len(q_text) > 30 and  # Reduced from 50 to capture more questions
                q_text[0].isupper() and 
                len(q_num.split('.')[1]) <= 2):  # Question number should be like 7.47, not 7.5394
                questions.append((q_num, q_text))
        
        print(f"Found {len(questions)} valid questions for Chapter {chap}")
        
        # Extract solutions from appendix A
        s_pattern = re.compile(
            rf"({chap}\.\d+)\s+(.*?)(?=\d+\.\d+|$)",
            re.MULTILINE | re.DOTALL
        )
        sol_matches = list(s_pattern.finditer(appendix_a))
        sol_map = {m.group(1): m.group(2) for m in sol_matches}
        print(f"Found {len(sol_map)} solutions for Chapter {chap}")
        
        # Match questions to solutions
        for q_num, q_text in questions:
            if q_num in sol_map:
                print(f"  Question {q_num}: has solution")
                q = clean_text(f"{q_num} {q_text}")
                a = clean_text(sol_map[q_num])
                qa_list.append((q, a))
            else:
                print(f"  Question {q_num}: no solution")

    # Write to JSONL file
    with jsonlines.open(output_path, mode='w') as writer:
        for ques, ans in qa_list:
            writer.write({"instruction": ques, "response": ans})

def topic_match(text):
    """Return True if text contains any CI or hypothesis-test keyword."""
    return bool(TOPIC_RE.search(text))

def clean_text(s):
    """Collapse whitespace & strip leading/trailing spaces."""
    return re.sub(r"\s+", " ", s).strip()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: extract_exercises.py <input.pdf> <output.jsonl>")
        sys.exit(1)
    extract_exercises(sys.argv[1], sys.argv[2])
    print(f"Extracted Exercises to {sys.argv[2]}")
