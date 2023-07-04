import ast
import glob
import os

import pandas as pd


def aggregate_files(all_files: List[str], aggregated_data_file: str):
    """Aggregates all files into one file.

    Args:
        all_files (List[str]): List of all files to be aggregated.
        aggregated_data_file (str): Path to the aggregated data file.
    """
    aggregated_snippets_data = pd.concat(
        (pd.read_csv(f) for f in all_files), ignore_index=True
    )
    aggregated_snippets_data.to_csv(aggregated_data_file, index=False)


def aggregate_snippets_data(
    snippets_data_path: str, aggregated_snippets_data_file: str
):
    """Aggregates all snippets data files into one file.

    Args:
        snippets_data_path (str): Path to the directory containing all snippets
           data files.
        aggregated_snippets_data_file (str): Path to the aggregated snippets
           data file.
    """
    all_files = glob.glob(os.path.join(snippets_data_path, "*.csv"))
    aggregate_files(all_files, aggregated_snippets_data_file)


def append_cast_snippets_partition_information(
    training_data_path: str, snippets_data_path: str
):
    """Appends snippets data with partition information.

    Args:
        training_data_path: Path to training data containing information about
           partitions.
        snippets_data_path: Path to the sentence-level snippets data.
    """
    training_dataset_df = pd.read_csv(training_data_path)
    snippets_dataset_df = pd.read_csv(snippets_data_path)

    snippets_data_partitions = []

    for _, sample in snippets_dataset_df.iterrows():
        query = sample["Input.query"]
        sentences = list(ast.literal_eval(sample["answerable_sentences"]))
        if len(sentences) == 0:
            sentences = list(ast.literal_eval(sample["unanswerable_sentences"]))
        training_input = query + " [SEP] " + sentences[0]
        training_samples = training_dataset_df[
            (training_dataset_df["sentences"] == training_input)
        ]
        partitions = set(training_samples["partition"])
        if len(partitions) > 1:
            if "train" in partitions:
                snippets_data_partitions.append("train")
            else:
                snippets_data_partitions.append("validation")
        else:
            snippets_data_partitions.append(partitions.pop())

    snippets_dataset_df["partition"] = snippets_data_partitions
    snippets_dataset_df.to_csv(snippets_data_path, index=False)


def append_cast_snippets_answerability_information(
    snippets_data_path: str, sentence_level_snippets_data_path: str
):
    """Appends the processes snippets data with answerability information.

    Args:
        snippets_data_path: Path to the original snippets data.
        sentence_level_snippets_data_path: Path to the sentence-level snippets
           data.

    Answerability information is based on the "no answer" option selected by
    annotators in CAsT-snippets. The value of "answerability" column equals to 0
    if no annotations were done for given passage. Otherwise, the value equals
    to 1. The value of "no_answer_annotations" column equals to the number of
    annotator that selected "no answer" option in the annotation process.
    """
    snippets_dataset_df = pd.read_csv(snippets_data_path)
    sentence_level_snippets_data_df = pd.read_csv(
        sentence_level_snippets_data_path
    )
    no_answer = {}
    no_answer_annotations = []

    for _, sample in snippets_dataset_df.iterrows():
        spans = ast.literal_eval(sample["text_spans_2"])
        if len(spans) == 0:
            key = (
                sample["Input.turn_id"]
                + "--"
                + sample["Input.passage_id"]
                + "--"
                + str(sample["Input.relevance_score"])
            )
            if key not in no_answer.keys():
                no_answer[key] = 1
            else:
                no_answer[key] = no_answer[key] + 1

    for _, sample in sentence_level_snippets_data_df.iterrows():
        key = (
            sample["Input.turn_id"]
            + "--"
            + sample["Input.passage_id"]
            + "--"
            + str(sample["Input.relevance_score"])
        )
        if key in no_answer.keys():
            no_answer_annotations.append(
                int(
                    no_answer[key]
                    if no_answer[key] <= 3
                    else no_answer[key] / 2
                )
            )
        else:
            no_answer_annotations.append(0)

    sentence_level_snippets_data_df[
        "no_answer_annotations"
    ] = no_answer_annotations
    sentence_level_snippets_data_df["answerability"] = [
        0 if x == 3 else 1 for x in no_answer_annotations
    ]
    sentence_level_snippets_data_df.to_csv(
        sentence_level_snippets_data_path, index=False
    )


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


def append_cast_unanswerable_with_query(
    unanswerable_data_path: str,
    snippets_data_path: str,
):
    """Appends CAsT unanswerable data with query.

    Args:
        unanswerable_data_path: Path to CAsT unanswerable data.
        snippets_data_path: Path to CAsT-snippets data.
    """
    dataset_df = pd.read_csv(unanswerable_data_path)
    snippets_data_df = pd.read_csv(snippets_data_path)

    queries = []

    for _, sample in dataset_df.iterrows():
        query = snippets_data_df[
            snippets_data_df["Input.turn_id"] == sample["turn_id"]
        ]["Input.query"].values[0]
        queries.append(query)

    dataset_df["query"] = queries
    dataset_df.to_csv(unanswerable_data_path, index=False)


def append_cast_unanswerable_with_partition_information(
    unanswerable_data_path: str, partition_information_path: str
):
    """Appends partition information to CAsT unanswerable data.

    Args:
        unanswerable_data_path: Path to CAsT unanswerable data.
        partition_information_path: Path to the file with training data.
    """
    unanswerable_data_df = pd.read_csv(unanswerable_data_path)
    partition_data_df = pd.read_csv(partition_information_path)

    query_partition = {}
    partitions = []

    for _, sample in partition_data_df.iterrows():
        query = sample["sentences"].split(" [SEP] ")[0]
        query_partition[query] = sample["partition"]

    for _, sample in unanswerable_data_df.iterrows():
        partitions.append(query_partition[sample["query"]])

    unanswerable_data_df["partition"] = partitions
    unanswerable_data_df.to_csv(unanswerable_data_path, index=False)
