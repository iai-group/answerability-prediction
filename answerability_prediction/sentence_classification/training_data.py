"""Functions for constructing training data for sentence-level answerability
classifier."""
import ast
import copy
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd
from datasets import load_dataset
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import train_test_split

from answerability_prediction.utils.data_processing import (
    aggregate_files,
    aggregate_snippets_data,
    append_cast_snippets_answerability_information,
    append_cast_snippets_partition_information,
    append_cast_unanswerable_with_partition_information,
    append_cast_unanswerable_with_query,
)


def extract_sentence_answerability_scores_from_snippets_data(
    snippets_data: pd.DataFrame, results_data_path: str
):
    """Extracts sentence-level answerability scores from the snippets data.

    Args:
        snippets_data (pd.DataFrame): The snippets data.
        results_data_path (str): Path to the resulting sentence-level
           answerability data.

    The resulting data contains separate columns for sentences that were
    selected by annotators in CAsT-snippets dataset and sentences that were not
    selected. Additionally, a column with sentence_key indentifiers is added
    which is a concatenation of turn_id and passage_id.
    """
    queries = defaultdict(set)
    passages = defaultdict(set)
    passages_ids = defaultdict(set)
    turn_ids = defaultdict(set)
    relevance_scores = defaultdict(set)
    text_spans = defaultdict(set)
    sentences_from_snippets = defaultdict(set)
    not_chosen_sentences = defaultdict(set)

    for _, row in snippets_data.iterrows():
        # print(row)
        key = row["Input.turn_id"] + "--" + row["Input.passage_id"]
        passage = row["Input.passage"]
        snippets = ast.literal_eval(row["text_spans_2"])

        turn_ids[key] = row["Input.turn_id"]
        queries[key] = row["Input.query"]
        passages[key] = row["Input.passage"]
        passages_ids[key] = row["Input.passage_id"]
        relevance_scores[key] = row["Input.relevance_score"]
        text_spans[key].union(snippets)

        sentences = sent_tokenize(passage)

        for sentence in sentences:
            sentence_no_whitespaces = (
                "".join(e for e in sentence if e.isalnum())
                .replace("/", "")
                .replace("-", "")
            )
            sentence_no_ascii = "".join(
                [i if ord(i) < 128 else "" for i in sentence_no_whitespaces]
            )
            sentence_is_chosen = False
            for snippet in snippets:
                snippet_sents = sent_tokenize(snippet)
                for snippet_sent in snippet_sents:
                    snippet_no_whitespaces = (
                        "".join(e for e in snippet_sent if e.isalnum())
                        .replace("/", "")
                        .replace("-", "")
                    )
                    snippet_no_ascii = "".join(
                        [
                            i if ord(i) < 128 else ""
                            for i in snippet_no_whitespaces
                        ]
                    )
                    if (
                        snippet_no_ascii in sentence_no_ascii
                        or sentence_no_ascii in snippet_no_ascii
                        or sentence_no_ascii == snippet_no_ascii
                    ):
                        sentence_is_chosen = True
            if sentence_is_chosen:
                sentences_from_snippets[key].add(sentence)
            else:
                not_chosen_sentences[key].add(sentence)

        not_chosen_sents_filtered = []
        for sent in not_chosen_sentences[key]:
            if sent not in sentences_from_snippets[key]:
                not_chosen_sents_filtered.append(sent)

        not_chosen_sentences[key] = set(not_chosen_sents_filtered)

        if len(sentences_from_snippets[key]) == 0:
            sentences_from_snippets[key] = set()
        if len(not_chosen_sentences[key]) == 0:
            not_chosen_sentences[key] = set()

    new_snippets_df = pd.DataFrame(
        {
            "Input.turn_id": turn_ids.values(),
            "Input.query": queries.values(),
            "Input.passage_id": passages_ids.values(),
            "Input.passage": passages.values(),
            "Input.relevance_score": relevance_scores.values(),
            "answerable_sentences": sentences_from_snippets.values(),
            "unanswerable_sentences": not_chosen_sentences.values(),
        }
    )

    new_snippets_df.to_csv(
        results_data_path, sep=",", encoding="utf-8-sig", index=False
    )


def prepare_snippets_training_data(
    snippets_data_df: pd.DataFrame, training_data_path: str
):
    """Prepares training data for answer-in-the-sentence classifier.

    Data from CAsT-snippets are split into partitions across topics not queries.
    Each data sample has a form 'query [SEP] sentence' where the label is 0 or
    1 depending on the sentence being selected by annotators or not.

    Args:
        snippets_data_df: The dataframe with sentence-level data from
           CAsT-snippets.
        training_data_path: Path to the directory where the training data will
           be stored.
    """
    sentences = {"train": [], "validation": [], "test": []}
    labels = {"train": [], "validation": [], "test": []}
    train_sentence_keys, test_sentence_keys, _, _ = train_test_split(
        list(set(snippets_data_df["Input.turn_id"])),
        list(set(snippets_data_df["Input.turn_id"])),
        test_size=0.1,
        random_state=42,
    )
    train_sentence_keys, val_sentence_keys, _, _ = train_test_split(
        train_sentence_keys,
        train_sentence_keys,
        test_size=0.11,
        random_state=42,
    )

    for _, sample in snippets_data_df.iterrows():
        if sample["Input.turn_id"] in train_sentence_keys:
            partition = "train"
        elif sample["Input.turn_id"] in val_sentence_keys:
            partition = "validation"
        else:
            partition = "test"
        query = sample["Input.query"]
        answerable_sentences = list(
            ast.literal_eval(sample["answerable_sentences"])
        )
        unanswerable_sentences = list(
            ast.literal_eval(sample["unanswerable_sentences"])
        )

        for sentence in answerable_sentences:
            sentences[partition].append("{} [SEP] {}".format(query, sentence))
            labels[partition].append(1)

        for sentence in unanswerable_sentences:
            sentences[partition].append("{} [SEP] {}".format(query, sentence))
            labels[partition].append(0)

    dataset = ["cast-snippets"] * (
        len(sentences["train"])
        + len(sentences["validation"])
        + len(sentences["test"])
    )
    partition = (
        ["train"] * len(sentences["train"])
        + ["validation"] * len(sentences["validation"])
        + ["test"] * len(sentences["test"])
    )

    training_dataset_df = pd.DataFrame(
        {
            "dataset": dataset,
            "partition": partition,
            "sentences": sentences["train"]
            + sentences["validation"]
            + sentences["test"],
            "labels": labels["train"] + labels["validation"] + labels["test"],
        }
    )
    training_dataset_df.to_csv(training_data_path)


def prepare_squad_training_data(
    training_data_path: str,
):
    """Preprocesses SQuAD 2.0 dataset to be used for unanswerability predicition.

    Args:
        training_data_path: Path to the directory where the training data will
            be stored.
    """

    squad = load_dataset("squad_v2")

    sentences = {
        "train": {
            "answerable_0": [],
            "answerable_1": [],
            "unanswerable": [],
        },
        "validation": {
            "answerable_0": [],
            "answerable_1": [],
            "unanswerable": [],
        },
    }
    labels = copy.deepcopy(sentences)

    for partition in ["train", "validation"]:
        for sample in squad[partition]:
            unanswerable = len(sample["answers"]["text"]) == 0
            query = sample["question"]
            text_sentences = sent_tokenize(sample["context"])

            if unanswerable:
                for sentence in text_sentences:
                    sentences[partition]["unanswerable"].append(
                        "{} [SEP] {}".format(query, sentence.replace("\n", ""))
                    )
                    labels[partition]["unanswerable"].append(0)
            else:
                for sentence in text_sentences:
                    sentence_sample = "{} [SEP] {}".format(
                        query, sentence.replace("\n", "")
                    )
                    contains_response = any(
                        snippet in sentence
                        for snippet in sample["answers"]["text"]
                    )
                    if contains_response:
                        sentences[partition]["answerable_1"].append(
                            sentence_sample
                        )
                        labels[partition]["answerable_1"].append(1)
                    else:
                        sentences[partition]["answerable_0"].append(
                            sentence_sample
                        )
                        labels[partition]["answerable_0"].append(0)

    for partition in ["train", "validation"]:
        missing_unanswerable = len(sentences[partition]["answerable_1"])
        missing_sentences, missing_labels = zip(
            *random.sample(
                list(
                    zip(
                        sentences[partition]["unanswerable"],
                        labels[partition]["unanswerable"],
                    )
                ),
                missing_unanswerable,
            )
        )
        sentences[partition] = sentences[partition]["answerable_1"] + list(
            missing_sentences
        )
        labels[partition] = labels[partition]["answerable_1"] + list(
            missing_labels
        )

    X_train, X_test, y_train, y_test = train_test_split(
        sentences["train"], labels["train"], test_size=0.1, random_state=42
    )

    dataset = ["squad"] * (
        len(X_train) + len(sentences["validation"]) + len(X_test)
    )
    partition = (
        ["train"] * len(X_train)
        + ["validation"] * len(sentences["validation"])
        + ["test"] * len(X_test)
    )

    training_dataset_df = pd.DataFrame(
        {
            "dataset": dataset,
            "partition": partition,
            "sentences": X_train + sentences["validation"] + X_test,
            "labels": y_train + labels["validation"] + y_test,
        }
    )
    training_dataset_df.to_csv(training_data_path)


def select_passages_with_low_relevance_scores(
    relevance_scores_path: str,
    number_of_passages_per_query: int,
    results_path: str,
):
    """Selects passages with low relevance scores from qrels.

    Args:
        relevance_scores_path: Path to the file with relevance scores.
        number_of_passages_per_query: The number of passages to select per query.
        results_path: Path to the results file.
    """
    relevance_scores_df = pd.read_csv(
        relevance_scores_path, sep="\t", encoding="utf-8"
    )

    turn_ids = list(relevance_scores_df["turn_id"].unique())
    selected_samples = pd.DataFrame()

    for turn_id in turn_ids:
        turn_id_qrels = relevance_scores_df[
            relevance_scores_df["turn_id"] == turn_id
        ]
        low_score_passages = turn_id_qrels[
            turn_id_qrels["relevance_score"] == 0
        ]
        selected_samples = pd.concat(
            [
                selected_samples,
                low_score_passages.sample(n=number_of_passages_per_query),
            ]
        )

    selected_samples.to_csv(results_path)


def prepare_unanswerable_cast_training_data(
    selected_passages: str, snippets_data: str, results_path: str
):
    """Prepares training data from CAsT unanswerable passages.

    Args:
        selected_passages: Passages with low relevance scores selected for every
           query.
        snippets_data: Path to CAsT-snippets data with information about
           partitions.
        results_path: Path to the results file.
    """
    passages_df = pd.read_csv(selected_passages)
    snippets_data_df = pd.read_csv(snippets_data)

    sentences = {"train": [], "validation": [], "test": []}
    labels = copy.deepcopy(sentences)
    turn_id_partition = {}

    for _, sample in snippets_data_df.iterrows():
        if sample["Input.turn_id"] not in turn_id_partition.keys():
            turn_id_partition[sample["Input.turn_id"]] = sample["partition"]

    for _, sample in passages_df.iterrows():
        partition = turn_id_partition[sample["turn_id"]]
        query = snippets_data_df[
            snippets_data_df["Input.turn_id"] == sample["turn_id"]
        ]["Input.query"].values[0]
        unanswerable_sentences = sent_tokenize(sample["passage"])

        for sentence in unanswerable_sentences:
            sentences[partition].append("{} [SEP] {}".format(query, sentence))
            labels[partition].append(0)

    dataset = ["cast-unanswerable"] * (
        len(sentences["train"])
        + len(sentences["validation"])
        + len(sentences["test"])
    )
    partition = (
        ["train"] * len(sentences["train"])
        + ["validation"] * len(sentences["validation"])
        + ["test"] * len(sentences["test"])
    )

    training_dataset_df = pd.DataFrame(
        {
            "dataset": dataset,
            "partition": partition,
            "sentences": sentences["train"]
            + sentences["validation"]
            + sentences["test"],
            "labels": labels["train"] + labels["validation"] + labels["test"],
        }
    )
    training_dataset_df.to_csv(results_path)


def read_training_data_from_df(
    training_data_df: pd.DataFrame,
) -> Tuple[Dict[str, List[str]], Dict[str, List[int]]]:
    """Reads training data from DataFrame into dictionaries of samples and labels.

    Args:
        training_data_path: DataFrame with training data.

    Returns:
        Dictionaries of samples and labels divided into partitions.
    """
    sentences = {"train": [], "validation": [], "test": []}
    labels = copy.deepcopy(sentences)

    for _, sample in training_data_df.iterrows():
        sentences[sample["partition"]].append(sample["sentences"])
        labels[sample["partition"]].append(sample["labels"])

    return sentences, labels


if __name__ == "__main__":
    # Prepare training data from CAsT-snippets
    snippets_data_sentence_answerability_path = (
        "data/CAsT-snippets/snippets_data_sentence-answerability.csv"
    )
    snippets_training_data_path = "data/CAsT-snippets/training_data.csv"
    years = ["2020", "2022"]
    for year in years:
        aggregate_snippets_data(
            "data/CAsT-snippets/{}".format(year),
            "data/CAsT-snippets/{}/{}_snippets_data.csv".format(year, year),
        )
        snippets_df = pd.read_csv(
            "data/CAsT-snippets/{}/{}_snippets_data.csv".format(year, year)
        )
        extract_sentence_answerability_scores_from_snippets_data(
            snippets_df,
            "data/CAsT-snippets/{}/{}_snippets_data_sentence-answerability.csv".format(
                year, year
            ),
        )

    aggregate_files(
        [
            "data/CAsT-snippets/2020/2020_snippets_data.csv",
            "data/CAsT-snippets/2022/2022_snippets_data.csv",
        ],
        "data/CAsT-snippets/snippets_data.csv",
    )
    aggregate_files(
        [
            "data/CAsT-snippets/2020/2020_snippets_data_sentence-answerability.csv",
            "data/CAsT-snippets/2022/2022_snippets_data_sentence-answerability.csv",
        ],
        snippets_data_sentence_answerability_path,
    )

    snippets_data_df = pd.read_csv(snippets_data_sentence_answerability_path)
    prepare_snippets_training_data(
        snippets_data_df, snippets_training_data_path
    )

    append_cast_snippets_partition_information(
        "data/CAsT-snippets/training_data.csv",
        snippets_data_sentence_answerability_path,
    )
    append_cast_snippets_answerability_information(
        "data/CAsT-snippets/snippets_data.csv",
        snippets_data_sentence_answerability_path,
    )

    # Prepare training data from SQuAD 2.0
    squad_training_data_path = "data/SQuAD-2/training_data.csv"
    prepare_squad_training_data(squad_training_data_path)

    # Prepare training data from unanswerable CAsT-snippets dataset extension
    # (CAsT-unanswerable)
    cast_unanswerable_data_path = "data/CAsT-unanswerable/unanswerable.csv"
    cast_unanswerable_training_data_path = (
        "data/CAsT-unanswerable/training_data.csv"
    )
    for year in years:
        select_passages_with_low_relevance_scores(
            "data/CAsT-unanswerable/{}_relevance_scores.csv".format(year),
            5,
            "data/CAsT-unanswerable/{}_unanswerable.csv".format(year),
        )

    aggregate_files(
        [
            "data/CAsT-unanswerable/2020_unanswerable.csv",
            "data/CAsT-unanswerable/2022_unanswerable.csv",
        ],
        cast_unanswerable_data_path,
    )
    prepare_unanswerable_cast_training_data(
        cast_unanswerable_data_path,
        snippets_data_sentence_answerability_path,
        cast_unanswerable_training_data_path,
    )

    append_cast_unanswerable_with_query(
        cast_unanswerable_data_path,
        snippets_data_sentence_answerability_path,
    )
    append_cast_unanswerable_with_partition_information(
        cast_unanswerable_data_path,
        cast_unanswerable_training_data_path,
    )

    aggregated_train_data = pd.concat(
        (
            pd.read_csv(f)
            for f in [snippets_training_data_path, cast_unanswerable_data_path]
        ),
        ignore_index=True,
    )
    aggregated_train_data.to_csv(
        "data/CAsT-answerability_training_data.csv", index=False
    )
