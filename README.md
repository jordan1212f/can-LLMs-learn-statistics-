# Can LLMs Learn Statistics?

This project investigates how well large language models (LLMs) can learn statistical reasoning and transfer that knowledge across related domains. We focus on two core tasks—confidence intervals and hypothesis testing—and compare three training methods:

2. **RAG (Retrieval-Augmented Generation)**  
   Inject relevant textbook context at inference time.  
3. **Hybrid (Fine-tuned + RAG)**  
   Combine both approaches for explanation quality and contextual grounding.

We evaluate open-source LLMs (e.g. LLaMA, DeepSeek, Mistral) on their ability to generalise step-by-step explanations, measure pedagogical clarity in line with Bloom’s 2-sigma problem, and ultimately assess their potential as AI tutors.

---
