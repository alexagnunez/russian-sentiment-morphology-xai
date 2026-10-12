# Does morphology-aware tokenisation make explanations easier to interpret?
### A Russian sentiment analysis case study

Standard Byte-Pair Encoding (BPE) splits words by frequency, not linguistic structure. In a morphologically rich language like Russian, this can break sentiment-bearing words into arbitrary fragments, and token-level explanation methods such as **SHAP** then explain predictions in terms of those fragments.

This project tests whether **linguistically informed tokenisation** produces more interpretable SHAP explanations, while keeping classification performance comparable.

## Method

1. **Data:** RuSentiment, a dataset of Russian social media posts (negative, neutral, positive and speech-act labels).
2. **Semi-supervised Morfessor:** trained on RuSentiment word frequencies plus SIGMORPHON gold morpheme boundaries, so segments are both domain-relevant and linguistically plausible.
3. **Two tokenisers with matched vocabulary size (20,626 tokens):**
   - standard SentencePiece BPE
   - morphology-aware: Morfessor segments kept whole, with BPE fallback only for unseen segments
4. **Two matched lightweight RoBERTa classifiers**, identical except for the tokeniser.
5. **Token-level SHAP** on Cyrillic tokens, with qualitative analysis of whether highly attributed tokens are recognisable, sentiment-relevant Russian units.

## Results

Morphological segmentation (held-out): **boundary F1 0.874**, exact match 0.852. Fewer than 0.5% of test segments needed BPE fallback.

| Split | Tokeniser | Accuracy | Macro F1 |
|---|---|---|---|
| Validation | Standard BPE | 0.7242 | 0.7100 |
| Validation | Morfessor + BPE fallback | 0.7336 | 0.7204 |
| Test | Standard BPE | 0.7741 | 0.7353 |
| Test | Morfessor + BPE fallback | 0.7669 | 0.7243 |

**Findings:** classification performance was broadly comparable. Morphology-aware tokens made SHAP explanations more meaningful when they preserved a sentiment-bearing unit that BPE had fragmented. The benefit depended on segmentation quality, though: where BPE already kept the word intact there was little gain, and Morfessor errors could make explanations less coherent.

## Run it

The notebook was built in Google Colab. Install the dependencies:

```
pip install -r requirements.txt
```

Then download the data and place the files in the notebook's working directory (they are not included in this repository):

- **RuSentiment** ([project page](https://text-machine.cs.uml.edu/lab2/projects/rusentiment/), [paper](https://aclanthology.org/C18-1064/), [GitHub](https://github.com/strawberrypie/rusentiment)): `rusentiment_random_posts.csv`, `rusentiment_preselected_posts.csv`, `rusentiment_test.csv`
- **SIGMORPHON** morpheme segmentation data, Russian, word level ([GitHub](https://github.com/sigmorphon)): `rus.word.train.tsv`, `rus.word.dev.tsv`, `rus.word.test.gold.tsv`

## Tools

Python · PyTorch · Hugging Face Transformers · SentencePiece · Morfessor · SHAP · scikit-learn · pandas

*Completed as part of the MSc Artificial Intelligence at Queen Mary University of London.*
