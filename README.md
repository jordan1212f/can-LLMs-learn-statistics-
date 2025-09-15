### ===SCRIPT EXPLANATIONS===

Judge_runner.py - A script used for running the judge. It's automated evaluation system calls to Qwen-7b (via Ollama) to grade the statistical analysis responses according to the rubric The script takes a JSONL file containing questions and model responses, loads a YAML-formatted rubric with scoring criteria, and systematically evaluates each response by rendering category specific prompts that are linked to the releative rubric sections (weights, deductions, applicable criteria). For each QA pair, it builds a context aware prompt by extracting only the rubric components relevant to that question's category, calls the Qwen model through the API with temperature=0 for consistent scoring, and expects structured JSON responses containing subscores, deductions, final scores, and rationale. It appends all this to a seperate JSONL file which can found in the folder QWEN_scores

Scorer.py - Post processing so that I didnt have to rely on Qwen to calculate the scores as LLMs tend to miscalcualte alot. The script loads JSONL files containing answer and the YAML rubric specification, then systematically applies deduction rules by mapping deduction codes to point values, enforcing category specific allowlists that determine which deductions are valid for each question type, and implementing scoring caps (both fractional limits and per-deduction maximums) to prevent excessive penalties. It processes each record by calculating final scores through a clamping function that ensures scores remain within the valid rubric interval of [0, 10].

You can find a  list of all files and datasets used under each specific folder

Questions that were editied to be more reasoning based: 
os_5.15c, os_5.16b, os_5.19c, os_5.25d, os_5.27, os_5.29f, full-test-procedure (reasoning) [All], Experiment design.

The RAG architecture is uploaded as a PDF, see Advanced_Rag.ipynb. The description of the architecutre is found within the dissertation itself.

