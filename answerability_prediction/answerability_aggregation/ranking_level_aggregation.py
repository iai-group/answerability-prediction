"""Functions for aggregating passage-level answerability scores on ranking-level."""

import argparse
import ast
import itertools
from typing import Dict

import pandas as pd
from sklearn.metrics import accuracy_score


def parse_args() -> argparse.Namespace:
    """Parses arguments from command-line call.

    Returns:
        Arguments from command-line call.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        help="The name of the sentence classifier to be used.",
        default="/snippets_unanswerable",
    )
    return parser.parse_args()


def aggregate_passage_probabilities_max_mean(
    passage_probabilities_file: str, results_file: str
) -> None:
    """Aggregates passage-level probabilities with max and mean.

    Args:
        passage_probabilities_file: File with passage-level answerability
           probabilities.
        results_file: File to save ranking-level aggregation results to.
    """
    passage_level_aggregation_results_df = pd.read_csv(
        passage_probabilities_file
    )
    unique_turn_ids = list(set(passage_level_aggregation_results_df["turn_id"]))
    ranking_level_aggregation_results = []

    for turn_id in unique_turn_ids:
        turn_id_samples = passage_level_aggregation_results_df[
            passage_level_aggregation_results_df["turn_id"] == turn_id
        ]
        passage_ids = list(turn_id_samples["passage_id"])
        passage_ids_triples = list(set(itertools.combinations(passage_ids, 3)))
        for passage_ids_triple in passage_ids_triples:
            answerabilities = []
            max_probs = []
            mean_probs = []
            passages = []
            for passage_id in passage_ids_triple:
                passage_sample = turn_id_samples[
                    turn_id_samples["passage_id"] == passage_id
                ]
                answerabilities.append(list(passage_sample["answerability"])[0])
                max_probs.append(list(passage_sample["max probability"])[0])
                mean_probs.append(list(passage_sample["avg probability"])[0])
                passages.append(list(passage_sample["passage"])[0])
            ranking_level_aggregation_results.append(
                {
                    "turn_id": turn_id,
                    "query": list(turn_id_samples["query"])[0],
                    "passage_ids": passage_ids_triple,
                    "passages": passages,
                    "partition": list(turn_id_samples["partition"])[0],
                    "answerabilities": answerabilities,
                    "max_probs_passage": max_probs,
                    "mean_probs_passage": mean_probs,
                    "max_max_prob_answer": max(max_probs),
                    "mean_max_prob_answer": sum(max_probs) / len(max_probs),
                    "max_mean_prob_answer": max(mean_probs),
                    "mean_mean_prob_answer": sum(mean_probs) / len(mean_probs),
                }
            )

    ranking_level_aggregation_results_df = pd.DataFrame(
        ranking_level_aggregation_results
    )
    ranking_level_aggregation_results_df.to_csv(results_file)


def test_man_mean_aggregation(
    aggregation_results_file: str,
    threshold_max: float,
    threshold_mean: float,
    partition: str,
) -> Dict[str, float]:
    """Evaluates max and mean aggregation on ranking-level.

    Args:
        aggregation_results_file: File with ranking-level aggregation results.
        threshold_max: Threshold for max aggregation.
        threshold_mean: Threshold for mean aggregation.
        partition: Partition to test on.

    Returns:
        Accuracy of max and mean aggregation on ranking-level.
    """
    ranking_level_aggregation_results_df = pd.read_csv(aggregation_results_file)
    predicted_labels_max_max = []
    predicted_labels_mean_max = []
    predicted_labels_max_mean = []
    predicted_labels_mean_mean = []
    ground_truth_labels = []

    for _, sample in ranking_level_aggregation_results_df.iterrows():
        if sample["partition"] == partition:
            predicted_labels_max_max.append(
                0 if sample["max_max_prob_answer"] < threshold_max else 1
            )
            predicted_labels_mean_max.append(
                0 if sample["mean_max_prob_answer"] < threshold_mean else 1
            )
            predicted_labels_max_mean.append(
                0 if sample["max_mean_prob_answer"] < threshold_max else 1
            )
            predicted_labels_mean_mean.append(
                0 if sample["mean_mean_prob_answer"] < threshold_mean else 1
            )
            ground_truth = (
                1 if 1 in ast.literal_eval(sample["answerabilities"]) else 0
            )
            ground_truth_labels.append(ground_truth)

    return {
        "max_max": accuracy_score(
            ground_truth_labels, predicted_labels_max_max
        ),
        "mean_max": accuracy_score(
            ground_truth_labels, predicted_labels_mean_max
        ),
        "max_mean": accuracy_score(
            ground_truth_labels, predicted_labels_max_mean
        ),
        "mean_mean": accuracy_score(
            ground_truth_labels, predicted_labels_mean_mean
        ),
    }


if __name__ == "__main__":
    args = parse_args()
    model_name = args.model_name

    cast_answerability_max_mean_aggregation = "data/aggregation_results/max_mean/passage/{}_classifier-CAsT_answerability.csv".format(
        model_name
    )
    ranking_level_cast_answerability_aggregation_results = "data/aggregation_results/max_mean/ranking/{}_classifier-CAsT_answerability.csv".format(
        model_name
    )

    aggregate_passage_probabilities_max_mean(
        cast_answerability_max_mean_aggregation,
        ranking_level_cast_answerability_aggregation_results,
    )

    max_threshold = 0.5
    mean_threshold = 0.25

    ranking_level_accuracy = test_man_mean_aggregation(
        ranking_level_cast_answerability_aggregation_results,
        max_threshold,
        mean_threshold,
        "test",
    )

    print(ranking_level_accuracy)
