# can-LLMs-learn-statistics-
This project explores how large language models generalise between related statistical tasks. We compare fine-tuning, RAG, and hybrid methods by training on confidence intervals and testing on hypothesis testing, using a range of models.

This project investigates how well large language models (LLMs) can learn to reason about statistical concepts — and whether they can transfer that reasoning across related domains.

Specifically, we fine-tune and evaluate open-source LLMs (e.g. LLaMA, DeepSeek, Mistral) using questions and answers about confidence intervals, and then test their ability to answer hypothesis testing questions — a closely related but distinct statistical domain.

Three different approaches are compared:

Fine-tuning only – aligns model reasoning through curated QA pairs

RAG (Retrieval-Augmented Generation) – injects context from a textbook corpus

Hybrid (Fine-tuned + RAG) – combines both approaches for explanation + retrieval

The goal is to measure how well each approach supports generalisation, step-by-step explanation, and pedagogical clarity — in line with Bloom’s 2-sigma problem and the broader potential of LLMs as AI tutors.

