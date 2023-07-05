"""Functions to aggregate sentence-level answerability scores on passage-level
and evaluate them in terms of accuracy."""

import argparse
import ast
from typing import List

import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from sklearn.metrics import accuracy_score
from transformers import BertForSequenceClassification, BertTokenizer

from answerability_prediction.sentence_classification.classifier import (
    get_sentence_classifier_logits_and_probabilities,
)


def get_sentence_level_answerability_prob_and_logits_for_dataset(
    dataset_path: str, results_path: str, model_name: str
):
    """Gets the sentence level answerability probabilities and logits for a dataset.

    Args:
        dataset_path (str): Path to the dataset for which sentence-level
           answerability scores are needed.
        results_path (str): Path to save the results file with sentence-level
           answerability scores.
        model_name (str): Name of the model to use.
    """
    model_save_path = (
        "models/answer_in_the_question/bert-base-uncased" + model_name
    )
    tuned_model = BertForSequenceClassification.from_pretrained(
        model_save_path + "/best_model"
    )
    tuned_tokenizer = BertTokenizer.from_pretrained(
        model_save_path + "/best_model"
    )

    dataset_df = pd.read_csv(dataset_path)
    testing_data_results = pd.DataFrame()

    for _, sample in dataset_df.iterrows():
        query = (
            sample["Input.query"]
            if "Input.query" in list(dataset_df.columns)
            else sample["query"]
        )
        passage = (
            sample["Input.passage"]
            if "Input.passage" in list(dataset_df.columns)
            else sample["passage"]
        )
        sentences = sent_tokenize(passage)
        sentences_probs = []
        sentences_logits = []
        for sentence in sentences:
            sentences_model_output = (
                get_sentence_classifier_logits_and_probabilities(
                    query + " [SEP] " + sentence,
                    tuned_model,
                    tuned_tokenizer,
                )
            )
            sentences_probs.append(sentences_model_output["prob"])
            sentences_logits.append(sentences_model_output["logits"])
        sample["sentence_probabilities"] = sentences_probs
        sample["sentence_logits"] = sentences_logits

        testing_data_results = testing_data_results.append(
            sample, ignore_index=True
        )

    testing_data_results.to_csv(results_path)


def aggregate_sentence_probabilities_max_mean(
    dataset_file: str, results_file: str
):
    """Aggregates sentence-level predictions for each query-passage pair.

    Args:
        dataset_file: File with data containing sentence-level answerability
           scores.
        results_file: File to save the passage-level aggregation results.
    """
    dataset_df = pd.read_csv(dataset_file)

    max_probs = []
    mean_probs = []

    for _, sample in dataset_df.iterrows():
        sentences_probabilities = [
            prob[1]
            for prob in ast.literal_eval(sample["sentence_probabilities"])
        ]
        max_prob = max(sentences_probabilities)
        avg_prob = sum(sentences_probabilities) / len(sentences_probabilities)
        max_probs.append(max_prob)
        mean_probs.append(avg_prob)

    dataset_df["max probability"] = max_probs
    dataset_df["avg probability"] = mean_probs

    dataset_df.to_csv(results_file)


def test_man_mean_aggregation(
    aggregation_results_file: str,
    threshold_max: float,
    threshold_mean: float,
    partition: str,
):
    """Tests max and mean aggregation methods on passage level.

    Args:
        aggregation_results_file: File with aggregation results.
        threshold_max: Threshold for max aggregation method.
        threshold_mean: Threshold for mean aggregation method.
        partition: Partition to be used for testing.

    Returns:
        Accuracy value for max and mean aggregation methods applied on passage
        level to aggregate sentence answerability scores.
    """
    aggregation_results_df = pd.read_csv(aggregation_results_file)
    predicted_labels_max = []
    predicted_labels_mean = []
    ground_truth_labels = []

    for _, sample in aggregation_results_df.iterrows():
        if sample["partition"] == partition or partition == "all":
            predicted_labels_max.append(
                0 if sample["max probability"] < threshold_max else 1
            )
            predicted_labels_mean.append(
                0 if sample["avg probability"] < threshold_mean else 1
            )
            if "answerability" not in list(aggregation_results_df.columns):
                ground_truth = 0
            else:
                ground_truth = sample["answerability"]
            ground_truth_labels.append(ground_truth)

    return {
        "max": accuracy_score(ground_truth_labels, predicted_labels_max),
        "mean": accuracy_score(ground_truth_labels, predicted_labels_mean),
    }


def merge_passage_level_aggregation_results(
    snippets_max_mean_aggregation_results_path: str,
    unanswerable_max_mean_aggregation_results_path: str,
    merged_results_path: str,
):
    """Merges passage level aggregation results into one file.

    Args:
        snippets_max_mean_aggregation_results_path: Path to CAsT-snippets
           aggregation results.
        unanswerable_max_mean_aggregation_results_path: Path to
           CAsT-unanswerable aggregation results.
        merged_results_path: Path to save the merged results.
    """
    snippets_aggregation_results_df = pd.read_csv(
        snippets_max_mean_aggregation_results_path
    )
    unanswerable_aggregation_results_df = pd.read_csv(
        unanswerable_max_mean_aggregation_results_path
    )

    columns_to_remove = [
        col
        for col in list(unanswerable_aggregation_results_df.columns)
        if "Unnamed" in col
    ]
    columns_to_remove.append("Q0")

    unanswerable_aggregation_results_df = (
        unanswerable_aggregation_results_df.drop(columns=columns_to_remove)
    )

    columns_to_remove = [
        col
        for col in list(snippets_aggregation_results_df.columns)
        if "Unnamed" in col
    ]
    columns_to_remove.extend(["answerable_sentences", "unanswerable_sentences"])
    snippets_aggregation_results_df = snippets_aggregation_results_df.drop(
        columns=columns_to_remove
    )
    snippets_aggregation_results_df.rename(
        columns={
            "Input.turn_id": "turn_id",
            "Input.query": "query",
            "Input.passage_id": "passage_id",
            "Input.passage": "passage",
            "Input.relevance_score": "relevance_score",
        },
        inplace=True,
    )

    unanswerable_aggregation_results_df["answerability"] = len(
        unanswerable_aggregation_results_df
    ) * [0]
    unanswerable_aggregation_results_df["no_answer_annotations"] = len(
        unanswerable_aggregation_results_df
    ) * [-1]

    merged_dataframes = pd.concat(
        [snippets_aggregation_results_df, unanswerable_aggregation_results_df],
        axis=0,
        ignore_index=True,
    )
    merged_dataframes.to_csv(merged_results_path)


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


if __name__ == "__main__":
    args = parse_args()
    model_name = args.model_name

    passage_aggregation_results_path = (
        "data/aggregation_results/max_mean/passage/"
    )

    cast_snippets = (
        "data/CAsT-snippets/snippets_data_sentence-answerability.csv"
    )
    cast_snippets_max_mean_aggregation = (
        passage_aggregation_results_path
        + "{}_classifier-CAsT_snippets.csv".format(model_name)
    )
    cast_snippets_sentence_predictions = "data/aggregation_results/{}_classifier-CAsT_snippets-sentence_predictions.csv".format(
        model_name
    )

    get_sentence_level_answerability_prob_and_logits_for_dataset(
        cast_snippets, cast_snippets_sentence_predictions, model_name
    )
    aggregate_sentence_probabilities_max_mean(
        cast_snippets_sentence_predictions, cast_snippets_max_mean_aggregation
    )

    cast_unanswerable_data = "data/CAsT-unanswerable/unanswerable.csv"
    cast_unanswerable_max_mean_aggregation = (
        passage_aggregation_results_path
        + "{}_classifier-CAsT_unanswerable.csv".format(model_name)
    )
    cast_unanswerable_sentence_predictions = "data/aggregation_results/{}_classifier-CAsT_unanswerable-sentence_predictions.csv".format(
        model_name
    )

    get_sentence_level_answerability_prob_and_logits_for_dataset(
        cast_unanswerable_data,
        cast_unanswerable_sentence_predictions,
        model_name,
    )
    aggregate_sentence_probabilities_max_mean(
        cast_unanswerable_sentence_predictions,
        cast_unanswerable_max_mean_aggregation,
    )

    cast_answerability_max_mean_aggregation = (
        passage_aggregation_results_path
        + "/{}_classifier-CAsT_answerability.csv".format(model_name)
    )
    merge_passage_level_aggregation_results(
        cast_snippets_max_mean_aggregation,
        cast_unanswerable_max_mean_aggregation,
        cast_answerability_max_mean_aggregation,
    )
    print(
        test_man_mean_aggregation(
            cast_answerability_max_mean_aggregation, 0.5, 0.25, "test"
        )
    )
