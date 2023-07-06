import argparse
import ast
import itertools
import time
from typing import Dict, List

import openai
import pandas as pd
import tiktoken
from sklearn.metrics import accuracy_score


def predict_answerability_score_with_chatgpt(
    chatgpt_version: str,
    sample_data_path: str,
    prompt: List[Dict[str, str]],
    results_file: str,
    passage_level: bool = False,
):
    """Predicts answerability scores for a sample of the dataset using ChatGPT.

    Args:
        chatgpt_version: Version of ChatGPT to use.
        sample_data_path: Path to the sample data.
        prompt: Prompt to use for ChatGPT.
        results_file: Path to the file to save the results to.
    """
    sample_data = pd.read_csv(sample_data_path)

    chatgpt_results = pd.DataFrame(
        columns=[
            "turn_id",
            "query",
            "passage_ids",
            "passages",
            "answerabilities",
            "predicted_value",
        ]
    )

    for _, sample in sample_data.iterrows():
        if sample["partition"] == "test":
            time.sleep(2)
            query = sample["query"]
            if passage_level:
                passages = sample["passage"]
                answerabilities = sample["answerability"]
            else:
                passages = " ".join(ast.literal_eval(sample["passages"]))
                answerabilities = sample["answerabilities"]
            input_sample = {
                "role": "user",
                "content": "Question: {} Passage: {}".format(query, passages),
            }
            if (
                num_tokens_from_messages(
                    prompt + [input_sample], model=chatgpt_version
                )
                > 4095
            ):
                predicted_response = "-1"
                predicted_value = 0
            else:
                response = openai.ChatCompletion.create(
                    model=chatgpt_version,
                    messages=prompt + [input_sample],
                )
                predicted_response = response["choices"][0]["message"][
                    "content"
                ]
                predicted_value = (
                    0
                    if (
                        "0" == predicted_response
                        or " 0" in predicted_response
                        or "no " in predicted_response
                        or "not " in predicted_response
                    )
                    else 1
                )

            sample_results = pd.DataFrame(
                {
                    "turn_id": [sample["turn_id"]],
                    "query": [sample["query"]],
                    "passage_ids": [sample["passage_ids"]],
                    "passages": [sample["passages"]],
                    "answerabilities": [answerabilities],
                    "predicted_response": [predicted_response],
                    "predicted_value": [predicted_value],
                }
            )
            chatgpt_results = chatgpt_results.append(
                sample_results, ignore_index=True
            )
            sample_results.to_csv(
                results_file, mode="a", index=False, header=False
            )


def num_tokens_from_messages(messages, model):
    """Returns the number of tokens used by a list of messages.

    Function inspired by
    https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4
        tokens_per_name = -1
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens


def aggregate_passage_level_results(
    passage_level_results: str, aggregated_results: str
):
    """Aggregates passage-level answerability predictions on ranking level.

    Args:
        passage_level_results: Path to the passage-level results.
        aggregated_results: Path to the aggregated results.
    """

    unique_turn_ids = list(set(passage_level_results["turn_id"]))
    answer_aggregation_results = []

    for turn_id in unique_turn_ids:
        if turn_id in list(passage_level_results["turn_id"]):
            turn_id_samples = passage_level_results[
                passage_level_results["turn_id"] == turn_id
            ]
            passage_ids = list(turn_id_samples["passage_id"])
            passage_ids_triples = list(
                set(itertools.combinations(passage_ids, 3))
            )
            for passage_ids_triple in passage_ids_triples:
                answerabilities = []
                predicted_values = []
                passages = []
                for passage_id in passage_ids_triple:
                    passage_sample = turn_id_samples[
                        turn_id_samples["passage_id"] == passage_id
                    ]
                    answerabilities.append(
                        list(passage_sample["answerability"])[0]
                    )
                    predicted_values.append(
                        list(passage_sample["predicted_value"])[0]
                    )
                    passages.append(list(passage_sample["passage"])[0])
                answer_aggregation_results.append(
                    {
                        "turn_id": turn_id,
                        "query": list(turn_id_samples["query"])[0],
                        "passage_ids": passage_ids_triple,
                        "passages": passages,
                        "answerabilities": answerabilities,
                        "avg_ans": sum(predicted_values)
                        / len(predicted_values),
                    }
                )

    answer_aggregation_results_df = pd.DataFrame(answer_aggregation_results)
    answer_aggregation_results_df.to_csv(aggregated_results)


def evaluate_chatgpt_results(
    chatgpt_results: pd.DataFrame, passage_level: bool = False
):
    """Evaluates ChatGPT results.

    Args:
        chatgpt_results: Passage or ranking level results.
        passage_level (optional): Specifies whether predictions are on 
           passage-level. Defaults to False.
    """
    if passage_level:
        predictions = [int(r) for r in list(chatgpt_results["predicted_value"])]
        ground_truth = chatgpt_results["answerability"]
    else:
        predictions = [int(r) for r in list(chatgpt_results["predicted_value"])]
        ground_truth = [
            1 if 1 in list(ast.literal_eval(answerability)) else 0
            for answerability in chatgpt_results["answerabilities"]
        ]
    return accuracy_score(ground_truth, predictions)


def parse_args() -> argparse.Namespace:
    """Parses arguments from command-line call.

    Returns:
        Arguments from command-line call.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zero_shot",
        action="store_true",
        help="Specifies the type of prompt. If true zero-shot variant is used.",
    )
    parser.add_argument(
        "--two_shot",
        action="store_true",
        help="Specifies the type of prompt. If true two-shot variant is used.",
    )
    parser.add_argument(
        "--passage_level",
        action="store_true",
        help="Specifies whether the predictions should be on passage level. By default, it is ranking-level.",
    )
    parser.add_argument(
        "--openai_organization",
        type=str,
        help="The ID of the organization linked to OpenAI account.",
    )
    parser.add_argument(
        "--openai_api_key",
        type=str,
        help="The OpenAI API key.",
    )
    parser.add_argument(
        "--regenerate_chatgpt_results",
        action="store_true",
        help="Specifies whether the results of ChatGPT should be regenerated.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cast_answerability_ranking_level = "data/aggregation_results/max_mean/ranking/squad_snippets_unanswerable_classifier-CAsT_answerability.csv"

    chatgpt_version = "gpt-3.5-turbo-0301"

    if args.zero_shot:
        prompt = [
            {
                "role": "system",
                "content": "You are an assistant verifying whether the question is answerable in the provided passage. Return 1 if the answer or partial answer to the question is provided in the passage and 0 otherwise. Return only a number without explanation.",
            },
        ]
        if args.passage_level:
            chatgpt_results_file = "data/aggregation_results/chatgpt/passage_level-cast_answerability-zero_shot.csv"
        else:
            chatgpt_results_file = "data/aggregation_results/chatgpt/ranking_level-cast_answerability-zero_shot.csv"
    elif args.two_shot:
        prompt = [
            {
                "role": "system",
                "content": "You are an assistant verifying whether the answer to the question is included in the provided text. Return 0 if the answer is not given in the text or 1 if the text containts answer to the question. Return only a number without explanaition.",
            },
            {
                "role": "user",
                "content": "Question: Why does waste compaction slow the biodegradation of organic waste? Text: Introduction. It is illegal to burn household or garden waste at home or in your garden. Burning waste is not only a nuisance to neighbours, it can release many harmful chemicals into the air you breathe.",
            },
            {"role": "assistant", "content": "0"},
            {
                "role": "user",
                "content": "Question: I remember Glasgow hosting COP26 last year, but unfortunately I was out of the loop. What was the conference about? Text: The 2021 United Nations Climate Change Conference, also known as COP26, is the 26th United Nations Climate Change conference. This conference will be the most important intergovernmental meeting on the climate crisis since the Paris agreement was passed in 2015. ",
            },
            {"role": "assistant", "content": "1"},
        ]
        chatgpt_results_file = "data/aggregation_results/chatgpt/ranking_level-cast_answerability-two_shot.csv"
        if args.passage_level and not args.regenerate_chatgpt_results:
            print(
                "There is no data generated in two-shot passage-level setting. The generation of ChatGPT results is required."
            )
            exit()

    openai.organization = args.openai_organization
    openai.api_key = args.openai_api_key

    squad_snippets_unans_data = pd.read_csv(cast_answerability_ranking_level)

    if args.regenerate_chatgpt_results:
        predict_answerability_score_with_chatgpt(
            chatgpt_version,
            cast_answerability_ranking_level,
            prompt,
            chatgpt_results_file,
        )
    chatgpt_results = pd.read_csv(chatgpt_results_file)

    print(evaluate_chatgpt_results(chatgpt_results, args.passage_level))

    if args.passage_level:
        aggregated_results_path = "data/aggregation_results/max_mean/ranking/ChatGPT-CAsT_answerability.csv"
        aggregate_passage_level_results(
            chatgpt_results, aggregated_results_path
        )

        ranking_aggregation_results_df = pd.read_csv(aggregated_results_path)

        predicted_labels_threshold_0_33 = []
        predicted_labels_threshold_0_66 = []
        ground_truth_labels = []

        for _, sample in ranking_aggregation_results_df.iterrows():
            predicted_labels_threshold_0_33.append(
                0 if sample["avg_ans"] < 0.33 else 1
            )
            predicted_labels_threshold_0_66.append(
                0 if sample["avg_ans"] < 0.66 else 1
            )
            ground_truth = (
                1
                if 1 in list(ast.literal_eval(sample["answerabilities"]))
                else 0
            )
            ground_truth_labels.append(ground_truth)

        print(
            {
                "avg_ans threshold 0.33": accuracy_score(
                    ground_truth_labels, predicted_labels_threshold_0_33
                ),
                "avg_ans threshold 0.66": accuracy_score(
                    ground_truth_labels, predicted_labels_threshold_0_66
                ),
            }
        )
