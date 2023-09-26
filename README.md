# Towards Reliable and Factual Response Generation: Detecting Unanswerable Questions in Information-seeking Conversations

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Summary

Generative AI models face the challenge of hallucinations that can undermine users' trust in such systems. We propose to approach the problem of conversational information seeking as a two-step process, where relevant passages in a corpus are identified first and then summarized into a final system response. This way we can automatically assess if the answer to the user's question is present in the corpus. Specifically, our proposed method employs a sentence-level classifier to detect if the answer is present, then aggregates these predictions on the passage level, and eventually across the top-ranked passages to arrive at a final answerability estimate. For training and evaluation, we develop a dataset based on the TREC CAsT benchmark that includes answerability labels on the sentence, passage, and ranking levels. We demonstrate that our proposed method represents a strong baseline and outperforms a state-of-the-art LLM on the answerability prediction task. 

## Data

The data used for answer-in-the-sentence classifier training and evaluation, as well as for evaluation of passage- and ranking-level answerability scores aggregation is covered in detail [here](data/README.md).

## Answerability Detection

The challenge of answerability in conversational information seeking arises from the fact that the answer is typically not confined to a single entity or text snippet, but rather spans across multiple sentences or even multiple paragraphs. 

At the core of our approach is a sentence-level classifier (more details [here](answerability_prediction/sentence_classification/README.md)) that can distinguish sentences that contribute to the answer from ones that do not. These sentence-level estimates are then aggregated on the passage level and then further on the ranking level (i.e., set of top-n passages) (more details [here](answerability_prediction/answerability_aggregation/README.md)) to determine whether the question is answerable. 

![alt text](system_architecture.png)

### ChatGPT

For reference, we compare against a state-of-the-art large language model (LLM), using the most recent snapshot of GPT-3.5 (gpt-3.5-turbo-0301) via the ChatGPT API. More details about the setup and implementation can be found [here](answerability_prediction/chatgpt/README.md). Data generated with ChatGPT are covered in details [here](data/README.md).

## Results

Results for answerability detection on the sentence-, passage-, and ranking-level in terms of classification accuracy. 

<table>
    <thead>
        <tr>
            <th colspan="2">Sentence</th>
            <th colspan="2">Passage</th>
            <th colspan="2">Ranking</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Classifier</td>
            <td>Accuracy</td>
            <td>Aggregation</td>
            <td>Accuracy</td>
            <td>Aggregation</td>
            <td>Accuracy</td>
        </tr>
        <tr>
            <td rowspan=4>CAsT-answerability</td>
            <td rowspan=4>0.752</td>
            <td rowspan=2>Max</td>
            <td rowspan=2>0.634</td>
            <td>Max</td>
            <td>0.790</td>
        </tr>
        <tr>
            <td>Mean</td>
            <td>0.891</td>
        </tr>
        <tr>
            <td rowspan=2>Mean</td>
            <td rowspan=2>0.589</td>
            <td>Max</td>
            <td>0.332</td>
        </tr>
        <tr>
            <td>Mean</td>
            <td>0.829</td>
        </tr>
        <tr>
            <td rowspan=4>CAsT-answerability + SQuAD 2.0</td>
            <td rowspan=4>0.779</td>
            <td rowspan=2>Max</td>
            <td rowspan=2>0.676</td>
            <td>Max</td>
            <td>0.810</td>
        </tr>
        <tr>
            <td>Mean</td>
            <td>0.848</td>
        </tr>
        <tr>
            <td rowspan=2>Mean</td>
            <td rowspan=2>0.639</td>
            <td>Max</td>
            <td>0.468</td>
        </tr>
        <tr>
            <td>Mean</td>
            <td>0.672</td>
        </tr>
        </tr>
            <td colspan=3 rowspan=2>ChatGPT (zero-shot)</td>
            <td rowspan=2>0.787</td>
            <td>T=0.33</td>
            <td>0.839</td>
        </tr>
         </tr>
            <td>T=0.66</td>
            <td>0.623</td>
        </tr>
        <tr>
            <td colspan=5>ChatGPT (zero-shot)</td>
            <td>0.669</td>
        </tr>
        </tr>
            <td colspan=5>ChatGPT (two-shot)</td>
            <td>0.601</td>
        </tr>
    </tbody>
</table>
