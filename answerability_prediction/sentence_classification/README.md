# Answer-in-the-Sentence Classifier

The answer-in-the-sentence classifier is trained on sentence-level data from the train portion of the CAsT-answerability dataset. In some of the experiments, this data is augmented by data from the SQuAD 2.0 dataset to provide the classifier with additional training material. More information about data used for training and evaluation of the classifier can be found [here](../../data/README.md). The implementation of the procedure used for processing training data can be found in [training_data.py](training_data.py). 

The classifier is built using a BERT transformer model with a sequence classification head on top BertForSequenceClassification provided by [HuggingFace](https://huggingface.co/docs/transformers/model\_doc/bert\#transformers.BertForSequenceClassification). Each data sample contains `question [SEP] sentence` as input and a binary answerability label. The output of the classifier is the probability that the sentence contains (part of) the answer to the question. The implementation of the answer-in-the-sentence classifier can be found in [classifier.py](classifier.py).

