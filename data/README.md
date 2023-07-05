# Data

## Sentence-level answerability data

This paper builds upon an existing dataset, referred to as **CAsT-snippets** (more information can be found [here](CAsT-snippets/README.md)). The CAsT-snippets dataset is built on the top-relevant passages and it is highly imbalanced in terms of answerable and unanswerable query-passage pairs. To address this issue we build a simulated unanswerable CAsT dataset, referred to as **CAsT-unanswerable** (more information can be found [here](CAsT-unanswerable/README.md)) that for each query from CAsT-snippets dataset contains randomly selected non-relevant passages. The resulting dataset, named **CAsT-answerability**, contains around 1.8k answerable and 1.9k unanswerable question-passage pairs. The CAsT-answerability dataset is divided into partitions on the question level to avoid information leakage. The data for the sentence-level answerability classifier is extracted from the CAsT-answerability dataset in such a way that each sentence overlapping with at least one information nugget annotation is labeled as 1 (containing answer) and the remaining sentences in the considered passage are labeled as 0 (not containing answer).

Additionally, we build sentence-level training data from the SQuAD 2.0 dataset[^1] to provide the classifier with additional training material and thus guidance in terms of questions that can be answered with a short snippet contained in a single sentence. Data from SQuAD 2.0 is downsampled to be balanced in terms of the number of answerable and unanswerable question-sentence pairs (training data built from SQuAD 2.0 can be found [here](SQuAD-2/training_data.csv)).

### Comparison of different datasets

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


### Statistics for the CAsT-answerability dataset
 
|  | answerable | unanswerable |
|---|---|---|
| question-sentence pairs (train+test) | 6,395 | 19,043 |
| question-passage pairs (train+test) | 1,778 | 1,932 |
| question-ranking pairs (test) | 4,035 | 504 |

## Sentence-level answerability aggregation data

### Passage-level

Aggregation methods applied to sentence-level answerability-scores result in passage-level answerability scores. 

The files with passage-level answerability scores can be found [here](aggregation_results/max_mean/passage/). The filenames contain information about the answer-in-the-sentence classifier (_squad_snippets_unanswerable_ or _snippets_unanswerable_) used for predicting sentence-level answerability scores and the dataset (_CAsT_answerablility_, _CAsT_snippets_, or _CAsT_unanswerable_) for which the aggregation methods are applied: `{model_name}_classifier-{dataset}`. The predictions returned by different classifiers for all the dataset can be found [here](aggregation_results/) with the filenames of the following format: ``{model_name}_classifier-{dataset}-sentence_predictions`.

Methods used for the generation of passage-level answerability scores can be found in [passage_level_aggregation.py](../answerability_prediction/answerability_aggregation/passage_level_aggregation.py). 


### Ranking-level

For ranking-level answerability, which is the ultimate task we are addressing, we consider different input rankings, i.e., sets of n=3 passages, for the same input question. Specifically, for each unique input test question from CAsT-answerability (38), we generate all possible n-element subsets of passages available for this question (both containing and not containing an answer), thereby simulating passage rankings of varying quality. These rankings represent inputs with various degrees of difficulty for the same question, ranging from all passages containing an answer to a single passage with an answer to _no answer found in the corpus_. This yields a total of 4.5k question-ranking pairs, of which 0.5k are unanswerable.

The files with ranking-level answerability scores can be found [here](aggregation_results/max_mean/ranking/). The filenames contain information about the answer-in-the-sentence classifier (_squad_snippets_unanswerable_ or _snippets_unanswerable_) used for predicting sentence-level answerability scores and the dataset (_CAsT_answerablility_) for which the aggregation methods are applied: `{model_name}_classifier-{dataset}`.

Methods used for the generation og ranking-level answerability scores can be found in [ranking_level_aggregation.py](../answerability_prediction/answerability_aggregation/ranking_level_aggregation.py). 


