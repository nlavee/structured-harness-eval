# glass/judges/prompts.py

AA_LCR_EQUALITY_V1 = """
Given the following question and gold answer, determine if the system's prediction is correct.
Question: {question}
Gold Answer: {gold_answer}
Prediction: {prediction}

Respond with exactly CORRECT or INCORRECT.
"""

NLI_SENTENCE_V1 = """
You are a fact-checking judge.
Context:
{context}

Sentence:
{sentence}

Determine if the sentence is supported by the context.
Respond with one of: SUPPORTED, CONTRADICTED, UNVERIFIED.
"""
