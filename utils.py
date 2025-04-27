# utils.py  (add somewhere near the other helpers)
import re

def extract_gold_answer(example, cfg):
    """
    Return the ground‑truth answer string for an example.

    cfg must contain the key that stores the gold answer, e.g.
        cfg["answer_key"] = "solution"   # or "answer"
    """
    raw = example[cfg["answer_key"]]
    # strip TeX boxing if present
    m = re.search(r"\\boxed\\{([^}]*)\\}", raw)
    return m.group(1).strip() if m else str(raw).strip()


EVAL_DATASETS = {
    # "AIME24": {"path": "math-ai/aime24", "answer_key": "solution", "question_key": "problem"},
    "MATH500": {"path": "HuggingFaceH4/MATH-500", "answer_key": "answer", "question_key": "problem"},
    "AMC23": {"path": "math-ai/amc23", "answer_key": "answer", "question_key": "question"},
    "OlympiadBench": {"path": "math-ai/olympiadbench", "answer_key": "answer", "question_key": "question"},
    # "GPQA": {"path": "math-ai/gpqa", "answer_key": "solution", "question_key": "problem"},
    "Minerva": {"path": "math-ai/minervamath", "answer_key": "answer", "question_key": "question"},
}

BASE_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"