# Aggregation of Sentence-level Answerability Scores

In reality, answers are not confined to a single sentence but can be spread across several passages. We thus need a method to aggregate results obtained from the sentence-level classifier to decide whether the question can be answered given (1) a particular passage or (2) a set of top-ranked passages, referred to as a ranking. We consider two simple aggregation functions, _max_ and _mean_. 

The aggregated answerability score for a given passage is compared against a fixed threshold; passages for which the aggregated value is higher than the threshold are classified as ones containing the answer. We set the values of the thresholds on the validation partition (11% of the train split) of the CAsT-answerability dataset (the values used are 0.5 for max and 0.25 for mean aggregation). Implementation of passage-level aggregation procedure can be found in [passage_level_aggregation.py](passage_level_aggregation.py). 

An analogous procedure is repeated for the top n=3 passages in the ranking to decide on ranking-level answerability. Here, the aggregation methods take the passage-level answerability scores as input (obtained using max or mean aggregation of sentence-level probabilities). The resulting values are compared against the fixed threshold (using the same values as for passage-level aggregation) to yield a final ranking-level answerability prediction. Implementation of the ranking-level aggregation procedure can be found in [ranking_level_aggregation.py](ranking_level_aggregation.py). 

Both passage- and ranking-level answerability scores are evaluated on the test partition of the CAsT-answerability dataset.  

## Passage-level answerability aggregation

The passage-level answerability predictions are evaluated on query-passage pairs where passages with at least one information nugget annotated are considered answerable. 

The results can be obtained by running this command:

`python -m answerability_prediction.answerability_aggregation.passage_level_aggregation --model_name {name_of_the_classifier}` 

Model name specifies the answer-in-the-sentence classifier used for generating sentence-level answerability scores and can be selected from: _snippets_unanswerable_ and _squad_snippets_unanswerable_.

## Ranking-level answerability aggregation

For ranking-level answerability, which is the ultimate task we are addressing, we consider different input rankings, i.e., sets of n=3 passages, for the same input question.

The results can be obtained by running this command:

`python -m answerability_prediction.answerability_aggregation.ranking_level_aggregation --model_name {name_of_the_classifier}` 

Model name specifies the answer-in-the-sentence classifier used for generating sentence-level answerability scores and can be selected from: _snippets_unanswerable_ and _squad_snippets_unanswerable_.