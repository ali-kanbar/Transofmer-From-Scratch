# Transformer from Scratch — IMDB Sentiment Classifier

This project is my hands-on attempt to understand how a Transformer actually works by building one piece by piece instead of relying on a pretrained model or a high-level Transformer implementation.

The final model is a small **encoder-only Transformer** trained to classify IMDB movie reviews as either **positive** or **negative**. It was built with PyTorch and trained from random initialization.

> In this project, “from scratch” means that I implemented the main Transformer architecture myself—including tokenization, multi-head self-attention, Transformer blocks, masking, embeddings, and the classification head. PyTorch is still used for tensors, automatic differentiation, and basic neural-network layers.

## Why I Built This

Transformers are often introduced through large libraries where most of the important work happens behind a single function call. That is useful when building applications, but it can make the architecture feel like a black box.

I built this project to answer practical questions such as:

- How does text become a sequence of numbers?
- Why do Transformers need positional embeddings?
- How are queries, keys, and values created?
- How does multi-head self-attention work internally?
- What is the difference between causal and bidirectional attention?
- How does a padding mask affect attention scores?
- How can one special token represent an entire review?
- What does a complete training loop for a Transformer look like?

The goal was not to create the most accurate sentiment model possible. The goal was to understand the full path from raw text to a trained prediction.

## What I Built

The project contains a complete sentiment-classification pipeline:

1. Download and prepare the IMDB review dataset.
2. Clean and tokenize raw review text.
3. Build a vocabulary from the training data.
4. Convert reviews into fixed-length token sequences.
5. Add `<pad>`, `<unk>`, and `<cls>` special tokens.
6. Create attention masks for padded positions.
7. Implement multi-head self-attention manually.
8. Build reusable Transformer encoder blocks.
9. Use the `<cls>` representation for classification.
10. Train the model with AdamW and learning-rate scheduling.
11. Save the best validation checkpoint in memory.
12. Run predictions on new, unseen sentences.

## Project Architecture

The model is intentionally small so that it is easier to understand and practical to train inside a notebook.

| Component | Configuration |
|---|---:|
| Vocabulary size | 8,000 tokens |
| Maximum review length | 200 tokens + 1 `<cls>` token |
| Embedding dimension | 64 |
| Attention heads | 4 |
| Transformer blocks | 2 |
| Feed-forward expansion | 4× embedding dimension |
| Output classes | 2 |
| Dropout | 0.1 |
| Total parameters | Approximately 0.63 million |

The model follows this general flow:

```text
Raw review
    ↓
Cleaning and tokenization
    ↓
Token IDs + padding mask
    ↓
Token embeddings + positional embeddings
    ↓
Transformer encoder blocks
    ↓
Final <cls> token representation
    ↓
Linear classification head
    ↓
Positive / Negative prediction
```

## How I Built It, Step by Step

### 1. Loading and Splitting the Data

I used the IMDB movie-review dataset, which contains roughly 50,000 labeled reviews split evenly between positive and negative sentiment.

After shuffling the rows with a fixed random seed, I separated the data into training and validation groups.

The notebook initially creates:

- 45,008 available training rows
- 5,000 available validation rows

For this experiment, I encoded and trained on a smaller subset:

- 8,000 training reviews
- 2,000 validation reviews

I used a subset to keep the experiment lightweight and make it possible to iterate without needing long training runs or specialized hardware. This also kept the focus on understanding the architecture rather than optimizing a large model.

### 2. Building a Simple Tokenizer

Before the model can process text, every review must be converted into numbers.

I created a simple tokenizer that:

- Removes HTML tags such as `<br />`
- Converts the text to lowercase
- Extracts words, numbers, and punctuation marks

For example:

```text
"This movie was amazing!"
```

becomes something similar to:

```text
["this", "movie", "was", "amazing", "!"]
```

This tokenizer is intentionally basic. It helped me understand the preprocessing pipeline without introducing the additional complexity of subword algorithms such as BPE or WordPiece.

### 3. Creating the Vocabulary

I counted token frequencies only in the training data and kept the most common tokens until the vocabulary reached 8,000 entries.

I also added three special tokens:

- `<pad>` — fills unused positions so every sequence has the same length
- `<unk>` — represents words that are not in the vocabulary
- `<cls>` — stores the sequence-level representation used for classification

Building the vocabulary from training data only is important because it prevents information from the validation set from leaking into the training pipeline.

### 4. Encoding Reviews

Each review is converted into token IDs and truncated to a maximum of 200 tokens. A `<cls>` token is then added to the beginning of the sequence.

Every final input therefore contains 201 positions:

```text
<cls> token_1 token_2 ... token_200
```

Shorter reviews are padded with `<pad>` tokens. At the same time, I create a mask containing:

- `1` for real tokens
- `0` for padding positions

This mask is later passed into self-attention so the model does not assign attention weight to meaningless padding.

### 5. Implementing Multi-Head Self-Attention

The central part of the project is the custom `MultiHeadSelfAttention` module.

The input representation is projected into three tensors:

- Queries (`Q`)
- Keys (`K`)
- Values (`V`)

Attention is calculated using:

```text
Attention(Q, K, V) = softmax(QKᵀ / √dₖ)V
```

I then split the embedding dimension across multiple attention heads. Each head can learn to focus on different relationships inside the review.

For example, one head may learn relationships involving negation, while another may focus on descriptive words or long-range connections between parts of a sentence.

After attention is calculated independently for each head, the results are combined and passed through a final projection layer.

### 6. Using Bidirectional Attention

This model is an encoder, not an autoregressive text generator. Because the entire movie review is available at once, a token is allowed to attend to tokens both before and after it.

That means the model does **not** use a causal mask.

The only mask used is the padding mask, which blocks attention to `<pad>` positions.

This was an important distinction for me to understand:

- A GPT-style decoder uses causal attention because it predicts future tokens one at a time.
- An encoder used for classification can use bidirectional attention because the full input is already known.

### 7. Building the Transformer Block

Each Transformer block contains:

1. Layer normalization
2. Multi-head self-attention
3. A residual connection
4. Another layer normalization
5. A feed-forward neural network
6. Another residual connection

The feed-forward network expands the embedding dimension by a factor of four, applies a GELU activation, and projects it back to the original dimension.

The block uses a pre-normalization design:

```python
x = x + attention(layer_norm(x))
x = x + feed_forward(layer_norm(x))
```

Residual connections help information and gradients move through the network, while layer normalization helps keep training stable.

### 8. Adding Token and Positional Embeddings

A Transformer does not automatically know token order. Without positional information, it would treat the same group of words similarly even when their order changes.

To solve this, I added two learned embeddings:

- Token embeddings represent the identity of each token.
- Positional embeddings represent where each token appears in the sequence.

The model adds them together before sending the sequence through the Transformer blocks.

```text
Input representation = token embedding + positional embedding
```

This allows the model to learn both what a token is and where it appears.

### 9. Turning the Encoder into a Classifier

After the sequence passes through both Transformer blocks, I take the final hidden representation of the first token—the `<cls>` token.

Because `<cls>` can attend to the entire review, its final representation can collect information from across the sequence.

That representation is passed through a linear layer that produces two logits:

```text
[negative score, positive score]
```

Cross-entropy loss is then used during training.

### 10. Training the Model

The model is trained using:

- AdamW optimizer
- Learning rate of `1e-3`
- Weight decay of `0.1`
- Cosine annealing learning-rate scheduler
- Batch size of `32`
- Gradient clipping with a maximum norm of `1.0`
- 1,000 training iterations

Every 100 steps, the notebook switches to evaluation mode and estimates validation loss and accuracy across several random validation batches.

I also keep a copy of the model state whenever validation accuracy improves. At the end of training, the best checkpoint is restored rather than simply using the final training step.

This helped me understand why a model’s last state is not always its best state. Validation performance can rise and fall during training, so checkpointing protects the strongest version seen during the run.

## Results

The recorded notebook run produced:

- **Approximately 0.63 million parameters**
- **Best sampled validation accuracy: 84.84%**
- Best checkpoint reached around training step 900
- Total recorded training time: approximately 422 seconds

Validation accuracy improved from roughly 46% before meaningful training to around 85% at its best point.

The model also correctly classified clear custom examples:

```text
"This movie was a complete waste of time..." → Negative
"An absolute masterpiece..."                → Positive
```

A neutral example was classified as negative with high confidence, which also exposes an important limitation: the model was trained using only two labels, so it has no true neutral category.

## What I Learned

### Attention is easier to understand when implemented directly

Before building it, multi-head attention felt abstract. Writing the tensor reshaping, matrix multiplication, scaling, masking, softmax, and head concatenation made the process much more concrete.

### Masking depends on the task

I learned that “the attention mask” is not always one universal mask. A classification encoder needs to ignore padding, while an autoregressive decoder also needs to hide future tokens.

### The `<cls>` token is learned, not magical

The `<cls>` token starts as a normal learned embedding. It becomes useful because the model repeatedly updates it through attention and is trained to use its final state for classification.

### Positional information is essential

Self-attention compares tokens directly but does not naturally preserve word order. Positional embeddings provide the sequence structure needed to understand phrases and sentence relationships.

### Data preparation is a large part of the model

Tokenization, vocabulary construction, unknown tokens, truncation, padding, and masks are not side details. They directly shape what the model can learn.

### Small models can still learn meaningful patterns

Even with only two Transformer blocks, 64-dimensional embeddings, and a limited training subset, the model learned useful sentiment patterns and reached respectable validation accuracy.

### Evaluation can be noisy

The notebook evaluates on random validation batches rather than the complete validation set. This is fast, but the reported accuracy is an estimate and can vary between evaluations.

### Building from scratch reveals hidden design decisions

Using a pretrained library often hides choices involving initialization, normalization order, masking, sequence length, optimizer settings, checkpointing, and pooling. Implementing the model exposed all of these decisions.

## Limitations

This project is designed for learning, so several parts are intentionally simple:

- The tokenizer is word-based and does not use subword tokenization.
- Rare and unseen words become `<unk>`.
- Reviews are truncated after 200 tokens.
- Only a subset of the available dataset is used for training.
- Validation accuracy is estimated from random batches instead of the full validation set.
- The model has only positive and negative labels.
- There is no dedicated test-set evaluation.
- The model and vocabulary are not saved to disk.
- The notebook downloads data directly from a remote URL.
- Training is not optimized for GPU execution or large-scale batching.

These limitations are acceptable for an educational implementation, but they would need to be addressed before treating the model as a production system.

## Possible Improvements

The next improvements I would explore are:

1. Train on the complete IMDB training set.
2. Evaluate on the entire validation and test sets.
3. Add GPU/device support.
4. Save and reload the model, vocabulary, and configuration.
5. Replace the basic tokenizer with BPE or WordPiece.
6. Use a proper `Dataset` and `DataLoader` pipeline.
7. Add early stopping based on validation loss.
8. Compare `<cls>` pooling with mean pooling.
9. Plot training and validation metrics.
10. Experiment with more layers, wider embeddings, and additional heads.
11. Add a neutral or multi-class sentiment dataset.
12. Visualize attention weights to inspect what the model focuses on.

## Running the Notebook

### Requirements

- Python 3.9+
- PyTorch
- Requests
- Jupyter Notebook or JupyterLab

The other imported modules are included in Python’s standard library.

Install the main dependencies with:

```bash
pip install torch requests notebook
```

Then launch Jupyter:

```bash
jupyter notebook
```

Open:

```text
Transformer_from_Scratch(1).ipynb
```

Run the cells in order from top to bottom. The notebook downloads the dataset automatically, builds the vocabulary, trains the model, and finishes with sample predictions.

## Main File

```text
Transformer_from_Scratch(1).ipynb
```

The notebook is organized as a learning path rather than only as a final implementation:

```text
Data → Tokenizer → Encoding → Attention → Transformer → Classifier → Training → Prediction
```

## Final Takeaway

This project helped me move from knowing the general idea of a Transformer to understanding how its individual parts work together in code.

The most valuable part was not the final accuracy. It was seeing the entire system connect: raw text becomes tokens, tokens become embeddings, self-attention builds contextual representations, the `<cls>` token summarizes the review, and training gradually turns those operations into a working classifier.

Building the architecture manually made the Transformer feel less like a black box and more like a collection of understandable design choices.
