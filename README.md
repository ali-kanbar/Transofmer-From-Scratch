# Sentiment Transformer (from scratch)

A small transformer, built from raw PyTorch (no pretrained models, no shortcuts), that reads a movie review and says positive or negative.

## What it does

You give it text like:

> "An absolute masterpiece, beautifully shot with a gripping story."

It reads the whole thing and predicts: **positive**.

## How it works, briefly

- **Tokenizer**: splits text into words, keeps the 8,000 most common ones, everything else becomes `<unk>`.
- **`<cls>` token**: a special token stuck at the front of every review. After the model reads the whole review, this token's output is used as a summary of the whole thing, that summary is what gets classified.
- **Attention**: every word looks at every other word in the review (not just the words before it, the whole review is available at once, unlike a text-generating model).
- **Classifier head**: takes that summary and outputs two numbers, how positive, how negative, and picks the higher one.

## Results

- 0.63M parameters
- Trained on 8,000 reviews, tested on 2,000
- ~5.5 minutes on a single CPU core
- **85.5% accuracy** on reviews it never saw during training

## Requirements

```
pip install torch requests
```

## How to run

Open `sentiment_transformer_from_scratch.ipynb` in Jupyter or Google Colab and run the cells top to bottom. It downloads the dataset itself, no manual setup needed.
