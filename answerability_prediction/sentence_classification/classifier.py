"""Functions for training sentence-level answer-in-the-sentence classifier."""
import argparse
from typing import Dict, List

import evaluate
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)

from answerability_prediction.sentence_classification.training_data import (
    read_training_data_from_df,
)

set_seed = 49


def compute_metrics(eval_pred):
    """Computes accuracy for the classifier.

    Args:
        eval_pred: Evaluation predictions.

    Returns:
        Accuracy score.
    """
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)


class SentenceClassifierDataset(torch.utils.data.Dataset):
    """Class for the dataset for the answer-in-the-sentence classifier."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx]) for key, val in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def fine_tune_model(
    checkpoint: str,
    model_save_path: str,
    sentences: Dict[str, List[str]],
    labels: Dict[str, List[int]],
):
    """Fine tunes BERT model for sentence-level answerability prediction.

    Args:
        checkpoint: Model checkpoint.
        model_save_path: Path to be used for saving fine-tuned model.
        sentences: Training samples.
        labels: Training labels.
    """
    tokenizer = BertTokenizer.from_pretrained(checkpoint, do_lower_case=True)

    train_encodings = tokenizer(
        sentences["train"], truncation=True, padding=True
    )
    val_encodings = tokenizer(
        sentences["validation"], truncation=True, padding=True
    )

    train_dataset = SentenceClassifierDataset(train_encodings, labels["train"])
    val_dataset = SentenceClassifierDataset(val_encodings, labels["validation"])
    training_args = TrainingArguments(
        output_dir=model_save_path,
        do_predict=True,
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate
        save_steps=100,
        save_total_limit=10,
        load_best_model_at_end=True,
        weight_decay=0.01,  # strength of weight decay
        logging_dir=model_save_path + "/logs",  # directory for storing logs
        learning_rate=5e-5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        seed=49,
    )
    model = BertForSequenceClassification.from_pretrained(
        checkpoint, num_labels=2
    ).to("cuda")

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(model_save_path + "/best_model")


def get_sentence_classifier_logits_and_probabilities(
    sentence: str,
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
) -> Dict[str, List[float]]:
    """Gets logits and probabilities for answer-in-the-sentence classifier.

    Args:
        sentence: Sentence to be classified.
        model: Model to be used for classification.
        tokenizer: Tokenizer for the model.

    Returns:
        Dictionary containing logits and probabilities returned by the sentence
        classifier.
    """
    test_enc = tokenizer(
        sentence, truncation=True, padding=True, return_tensors="pt"
    )
    predictions = model(**test_enc)
    preds = nn.functional.softmax(predictions.logits, dim=-1)
    return {"prob": preds.tolist()[0], "logits": predictions.logits.tolist()[0]}


def test_sentence_classifier(
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    test_data: List[str],
    test_labels: List[int],
    threshold: float,
) -> float:
    """Tests answer-in-the-sentence classifier.

    Args:
        model: The sentence classifier.
        tokenizer: Tokenizer for the sentence classifier.
        test_data: Test data samples.
        test_labels: Ground truth test labels.
        threshold: The value of the threshold for answerability scores.

    Returns:
        Accuracy score of the answer-in-the-sentence classifier.
    """
    predicted_labels = []
    for sample in test_data:
        prediction = get_sentence_classifier_logits_and_probabilities(
            sample, model, tokenizer
        )["prob"]
        label = 0 if prediction[0] > threshold else 1
        predicted_labels.append(label)

    return accuracy_score(test_labels, predicted_labels)


def parse_args() -> argparse.Namespace:
    """Parses arguments from command-line call.

    Returns:
        Arguments from command-line call.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        help="The name of the sentence classifier.",
        default="/snippets_unanswerable",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Specifies that the classifier is to be tested.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Specifies that the classifier is to be trained.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Threshold for answerability scores returned by sentence classifier.",
        default=0.5,
    )
    return parser.parse_args()


def main(args: argparse.Namespace, checkpoint: str, model_save_path: str):
    """Main function for training and/or testing sentence classifier.

    Args:
        args: Arguments from command-line call.
        checkpoint: Model checkpoint.
        model_save_path: Path to be used for saving fine-tuned model.
    """
    if args.train:
        train_data_files = []
        if "snippets" in args.model_name:
            train_data_files.append("data/CAsT-snippets/training_data.csv")
        if "unanswerable" in args.model_name:
            train_data_files.append("data/CAsT-unanswerable/training_data.csv")
        if "squad" in args.model_name:
            train_data_files.append("data/SQuAD-2/training_data.csv")

        print(train_data_files)
        aggregated_train_data = pd.concat(
            (pd.read_csv(f) for f in train_data_files), ignore_index=True
        )
        sentences_train, labels_train = read_training_data_from_df(
            aggregated_train_data
        )
        fine_tune_model(checkpoint, model_save_path, sentences_train, labels_train)

    if args.test:
        tuned_model = BertForSequenceClassification.from_pretrained(
            model_save_path
        )
        tuned_tokenizer = BertTokenizer.from_pretrained(model_save_path)
        test_data_df = pd.read_csv("data/CAsT-answerability_training_data.csv")
        test_sentences, test_labels = read_training_data_from_df(test_data_df)
        print(
            "Model accuracy: ",
            test_sentence_classifier(
                tuned_model,
                tuned_tokenizer,
                test_sentences["test"],
                test_labels["test"],
                threshold=args.threshold,
            ),
        )

if __name__ == "__main__":
    args = parse_args()

    checkpoint = "bert-base-uncased"
    model_save_path = "models/" + args.model_name + "/best_model"

    main(args, checkpoint, model_save_path)