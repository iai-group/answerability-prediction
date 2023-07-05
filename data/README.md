# Data

## Sentence-level answerability data

This paper builds upon an existing dataset, referred to as **CAsT-snippets** (more information can be found [here](CAsT-snippets/README.md)). The CAsT-snippets dataset is built on the top-relevant passages and it is highly imbalanced in terms of answerable and unanswerable query-passage pairs. To address this issue we build a simulated unanswerable CAsT dataset, referred to as **CAsT-unanswerable** (more information can be found [here](CAsT-unanswerable/README.md)) that for each query from CAsT-snippets dataset contains randomly selected non-relevant passages. The resulting dataset, named **CAsT-answerability**, contains around 1.8k answerable and 1.9k unanswerable question-passage pairs. Additionally, we build sentence-level training data from the SQuAD 2.0 dataset[^1] to provide the classifier with additional training material and thus guidance in terms of questions that can be answered with a short snippet contained in a single sentence. Data from SQuAD 2.0 is downsampled to be balanced in terms of the number of answerable and unanswerable question-sentence pairs (training data built from SQuAD 2.0 can be found [here](SQuAD-2/training_data.csv)).

Comparison of different datasets:
|  | SQuAD 2.0 | CAsT-snippets | CAsT-unanswerable |
|---|---|---|---|
| questions | 142,192 | 371 | 371 |
| ans. questions | 92,749 | 365 | 0 |
| unans. questions | 49,443 | 6 | 371 |
| query-passage pairs | 142,192 | 1,855 | 1,855 |
| ans. query-passage pairs | 92,749 | 1778 | 0 |
| unans. query-passage pairs | 49,443 | 77 | 1,855 |
| sentences w/ answers in ans. query-passage pairs | 106,146 | 6395 | 0 |
| sentences w/o answers in ans. query-passage pairs | 369,025 | 5839 | 0 |
| sentences w/o answers in unans. query-passage pairs | 250,365 | 453 | 12,751 |


Statistics for the CAsT-answerability dataset: 
|  | answerable | unanswerable |
|---|---|---|
| question-sentence pairs (train+test) | 6,395 | 19,043 |
| question-passage pairs (train+test) | 1,778 | 1,932 |
| question-ranking pairs (test) | 4,035 | 504 |
