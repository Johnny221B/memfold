"""Prepare the official LongMemEval GPT-4o judge prompts as Batch JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def prompt(task: str, question: str, answer: str, response: str, abstention: bool) -> str:
    if abstention:
        template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
    elif task in {"single-session-user", "single-session-assistant", "multi-session"}:
        template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
    elif task == "temporal-reasoning":
        template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
    elif task == "knowledge-update":
        template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
    elif task == "single-session-preference":
        template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
    else:
        raise ValueError(task)
    return template.format(question, answer, response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    predictions = {row["id"]: row for row in map(json.loads, args.predictions.open())}
    references = {row["question_id"]: row for row in json.loads(args.references.read_text())}
    if set(predictions) != set(references) or len(predictions) != 500:
        raise ValueError("judge inputs must match all 500 references exactly")
    with args.output.open("w") as handle:
        for qid, ref in references.items():
            body = {"model": "gpt-4o-2024-08-06", "messages": [{"role": "user", "content": prompt(
                ref["question_type"], ref["question"], ref["answer"], predictions[qid]["prediction"], qid.endswith("_abs")
            )}], "temperature": 0, "max_tokens": 10}
            handle.write(json.dumps({"custom_id": qid, "method": "POST", "url": "/v1/chat/completions", "body": body}) + "\n")
    print(json.dumps({"requests": 500, "judge": "gpt-4o-2024-08-06", "prompt": "official_longmemeval"}))


if __name__ == "__main__": main()
