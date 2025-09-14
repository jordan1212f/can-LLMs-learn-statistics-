#!/usr/bin/env python3
"""
eextract questions only from statistics_eval_questions.jsonl
this script creates a new json file containing only questions and remove ansewres
"""

import json
import sys
from pathlib import Path

def extract_questions_only(input_file, output_file):
    questions_only = []
    
    print(f"Reading JSONL file: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        line_number = 0
        for line in f:
            line_number += 1
            line = line.strip()
            
            #Skip empty lines and comments
            if not line or line.startswith('\\'):
                continue
            
            try:
                # Parse JSON object
                json_obj = json.loads(line)
                
                # Create new object with only question_id, question, and category
                question_only = {}
                if 'question_id' in json_obj:
                    question_only['question_id'] = json_obj['question_id']
                if 'question' in json_obj:
                    question_only['question'] = json_obj['question']
                if 'category' in json_obj:
                    question_only['category'] = json_obj['category']
                
                # Only add if we have at least a question
                if 'question' in question_only:
                    questions_only.append(question_only)
                
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse JSON on line {line_number}: {e}")
                continue
    
    if not questions_only:
        print("No valid questions found in the file!")
        return
    
    print(f"Found {len(questions_only)} questions")
    
    # Write questions-only file
    print(f"Writing questions-only file: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for question_obj in questions_only:
            json.dump(question_obj, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"Successfully extracted {len(questions_only)} questions to new file")

def main():
    # Define input and output file paths
    input_file = "/Users/jordanfernandes/Desktop/Dissertation workspace/statistics_eval_questions.jsonl"
    output_file = "/Users/jordanfernandes/Desktop/Dissertation workspace/stats_eval_questions_only.jsonl"
    
    # Check if input file exists
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' not found!")
        sys.exit(1)
    
    try:
        extract_questions_only(input_file, output_file)
        print(f"\nExtraction completed successfully!")
        print(f"Original file: {input_file}")
        print(f"Questions-only file: {output_file}")
        
    except Exception as e:
        print(f"Error during extraction: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
