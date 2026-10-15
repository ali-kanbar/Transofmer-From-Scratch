# Transformer from Scratch

A small transformer that classifies movie reviews as **positive** or **negative**.

The tokenizer, self-attention and encoder blocks are written in Python and PyTorch. The model is trained from scratch on IMDB reviews, without pretrained weights or `nn.Transformer`.

## How it works

1. Clean the review, lowercase it and split it into tokens.
2. Convert the tokens to IDs, add `<cls>` at the beginning and pad shorter reviews.
3. Add token and position embeddings, then pass them through two transformer blocks.
4. Use the final `<cls>` vector to predict sentiment. Softmax converts the two output scores into probabilities.

Attention can look at the whole review in both directions. Padding positions are masked out.

| Setting | Value |
| --- | --- |
| Vocabulary limit | 8,000 tokens, including special tokens |
| Review length | Up to 200 tokens, plus `<cls>` |
| Embedding size | 64 |
| Transformer blocks | 2 |
| Attention heads per block | 4 |
| Dropout | 0.1 |

## Examples

**A positive sentence**

<img width="807" height="443" alt="image" src="https://github.com/user-attachments/assets/e6447422-0b1a-4c74-8c47-3d1a4fbab706" />

**A mixed sentence**

<img width="862" height="443" alt="image" src="https://github.com/user-attachments/assets/42f254b4-0fe5-451f-be4b-b6fbfea9d295" />

The second example shows a limitation: a fairly neutral sentence gets a strongly negative prediction. There is no neutral class, and a high model probability does not guarantee a correct answer.

## Training

The notebook downloads the [IMDB review CSV](https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv), shuffles the data and builds the vocabulary from the training split.

To keep the experiment small, training uses 8,000 reviews and a validation subset of 2,000 reviews. It runs for 1,000 updates with a batch size of 32, AdamW and a starting learning rate of `1e-3`.

Validation is checked on 10 random batches every 100 steps. The best weights are kept in memory and restored at the end. This is a sampled validation score, not accuracy on a separate test set.

## Run it

Install the packages for a local Jupyter session:

```bash
python -m pip install torch requests matplotlib notebook
jupyter notebook Transformer_from_Scratch.ipynb
```

You can also open the notebook in Google Colab. Run the cells from top to bottom; the dataset downloads automatically.

After training, try your own sentence:

```python
label, probability = predict("I enjoyed the movie")
print(f"{label}: {probability:.1%}")
```

For the plots, place `sentiment_visualization.py` beside the notebook and run this in a new notebook cell:

```python
%run -i sentiment_visualization.py
```

It asks for a sentence and saves the architecture diagram and prediction chart in `images/`. To save another example:

```python
result = visualize_sentiment(
    "I enjoyed the movie",
    filename="images/positive-example.png"
)
```

The model is trained on movie reviews, so results on other kinds of text may be less reliable. Long inputs are cut to the first 200 tokens, and words outside the vocabulary become `<unk>`.
