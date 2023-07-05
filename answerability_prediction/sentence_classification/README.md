# Answer-in-the-Sentence Classifier

The answer-in-the-sentence classifier is trained on sentence-level data from the train portion of the CAsT-answerability dataset. In some of the experiments, this data is augmented by data from the SQuAD 2.0 dataset to provide the classifier with additional training material. More information about data used for training and evaluation of the classifier can be found [here](../../data/README.md). The implementation of the procedure used for processing training data can be found in [training_data.py](training_data.py). 

The classifier is built using a BERT transformer model with a sequence classification head on top BertForSequenceClassification provided by [HuggingFace](https://huggingface.co/docs/transformers/model\_doc/bert\#transformers.BertForSequenceClassification). Each data sample contains `question [SEP] sentence` as input and a binary answerability label. The output of the classifier is the probability that the sentence contains (part of) the answer to the question. The implementation of the answer-in-the-sentence classifier can be found in [classifier.py](classifier.py).

Sentence-level answerability prediction is evaluated on the test partition of the CAsT-answerability dataset. The results can be obtained by running this command:
`python -m answerability_prediction.sentence_classification.classifier --model_name {name_of_the_classifier} --test` 

The name of the classifier (_name_of_the_classifier_) can be selected from the following list:
  - [snippets_unanswerable](../../models/snippets_unanswerable/) - BERT model trained for answerability prediction with train partition of CAsT-snippets data extended with CAsT-unanswerable (CAsT-answerability)
  - [squad_snippets_unanswerable](../../models/squad_snippets_unanswerable/) - BERT model trained for answerability prediction with train partition of CAsT-answerability dataset extended with downsampled SQuAD 2.0

It is also possible to fine-tune a new classifier with the following command:
`python -m answerability_prediction.sentence_classification.classifier --model_name {name_of_the_classifier} --train --test` 

The default value of the answerability score threshold is 0.5. The value can be changed by adding additional command line argument (`--threshold {threshold_value}`).