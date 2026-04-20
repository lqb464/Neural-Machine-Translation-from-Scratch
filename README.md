# RNN-With-Attention-For-Machine-Translation-From-Scratch

Neural Machine Translation (NMT) project from English to Vietnamese using RNN with Attention Mechanism, built entirely from scratch (embedding, GRU, encoder, decoder, attention, optimizer, scheduler, dataloader).

## Key Features

- **From Scratch**: Self-implemented core components (GRU, attention, embedding, activations).
- **Multiple Attention**: Bahdanau (additive) and Luong (multiplicative) attention.
- **Optimizers**: Adam, AdamW (with group weight decay), SGD.
- **Tokenization**: Basic or BPE (Byte-Pair Encoding) for subword handling.
- **Flexible Limiting**: Percentile-based token limits (adapts to data) + fixed fallback.
- **Experiment Pipeline**: Config-driven, CLI overrides, checkpointing, metrics (BLEU, ROUGE-L, chrF), logging, visualization.

## Directory Structure

```text
RNN-With-Attention-For-Machine-Translation-From-Scratch/
├── configs/
│   └── default.yaml                  # Hyperparameters and config (attention_type, optimizer, etc.)
├── data/
│   ├── en_sents                      # EN data (one sentence per line)
│   └── vi_sents                      # VI data (one sentence per line)
├── outputs/
│   └── seq2seq_attention/            # Checkpoint, vocab, history, plots after training
├── scripts/
│   ├── train.py                      # Train model with config + CLI override
│   ├── evaluate.py                   # Evaluate checkpoint on test set
│   ├── translate.py                  # Translate one sentence from EN to VI
│   └── test.py                       # Alias for evaluate.py
├── src/
│   ├── data/
│   │   ├── dataset.py                # Dataset with percentile limiting
│   │   ├── dataloader.py             # Custom DataLoader
│   │   ├── tokenizer.py              # Tokenizer basic/BPE
│   │   └── vocabulary.py             # Vocab building
│   ├── models/
│   │   ├── activations.py            # Activations (softmax, tanh)
│   │   ├── attention.py              # Bahdanau & Luong attention
│   │   ├── build.py                  # Model builder with attention type
│   │   ├── decoder.py
│   │   ├── encoder.py
│   │   ├── layers.py                 # Embedding, GRU
│   │   └── seq2seq.py
│   ├── training/
│   │   ├── evaluate.py               # Evaluator with metrics
│   │   ├── losses.py
│   │   ├── metrics.py                # BLEU, ROUGE-L, chrF
│   │   ├── optimizers.py             # Adam, AdamW, SGD (AdamW uses torch.optim)
│   │   ├── schedulers.py
│   │   ├── trainer.py                # Training loop with checkpoint
│   │   └── visualize.py
│   ├── utils/
│   │   ├── io.py                     # Load data/config, filter length
│   │   └── seed.py                   # Seed setup
│   └── factories.py                  # Centralized component building
├── requirements.txt
├── README.md
```

## Installation and Running

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
- Place data files in `data/`:
  - `en_sents`: English sentences (one sentence per line).
  - `vi_sents`: Corresponding Vietnamese sentences.
- Data will be automatically split into train/val in the code.

### 3. Train Model
```bash
# Train with default config (AdamW, Bahdanau, BPE)
python scripts/train.py

# Override config via CLI
python scripts/train.py --epochs 10 --batch_size 32 --lr 0.0005 --attention_type luong

# Train without attention (baseline)
python scripts/train.py --no_attention
```

- Outputs will be saved in `outputs/seq2seq_attention/` (checkpoints, vocab, history.json, plots).

### 4. Evaluate Model
```bash
# Evaluate latest checkpoint
python scripts/evaluate.py

# Evaluate specific checkpoint
python scripts/evaluate.py --checkpoint outputs/seq2seq_attention/best_checkpoint.pt
```

- Prints BLEU, ROUGE-L, chrF scores.

### 5. Translate Sentence
```bash
# Translate one sentence
python scripts/translate.py --sentence "Hello world"

# Translate with specific checkpoint
python scripts/translate.py --sentence "How are you?" --checkpoint outputs/seq2seq_attention/best_checkpoint.pt
```

## Config Options

You can edit `configs/default.yaml` to customize:

- **Model**: `attention_type` (bahdanau/luong), `embedding_dim`, `hidden_size`.
- **Training**: `optimizer` (adam/adamw/sgd), `lr`, `batch_size`, `epochs`.
- **Data**: `src_tokenizer` (basic/bpe), `max_src_len_percentile` (0.95), `max_src_len` (80).
- **Scheduler**: `type` (step/cosine), `warmup_steps`.

## Example Results

After training, model achieves ~20-30 BLEU on test set (depends on data). Check `outputs/seq2seq_attention/train_history.json` and plots.

## Notes

- Project maintains "from scratch" spirit but uses torch.optim for AdamW for efficiency.
- Percentile limiting helps avoid OOM on long data.
- BPE tokenization improves handling of rare words.

If errors occur, check data paths and torch version.

## How to Run

```bash
pip install -r requirements.txt
python scripts/train.py --config configs/default.yaml
```

Quick run for smoke test:

```bash
python scripts/train.py --epochs 1 --num_samples 100 --batch_size 8 --output_dir outputs/smoke_test
```

Translate after training:

```bash
python scripts/translate.py --sentence "i love you"
```

Evaluate checkpoint:

```bash
python scripts/evaluate.py --max_samples 200
```

## Notes

- `outputs/*/vocab/*.pkl` are created after training, used for translate/evaluate.
- `outputs/*/resolved_config.json` saves the actual config after CLI overrides.
- `outputs/*/train_history.csv` and `train_history.json` are used to compare runs.
