# Domain-Specific BPE Tokenizers for NER

This project implements custom Byte Pair Encoding (BPE) tokenizers for Named Entity Recognition across multiple text domains.

Three tokenizers were developed:

- **Domain 1:** optimized for noisy social-media text by normalizing URLs, mentions, hashtags, emojis, and repeated characters.
- **Domain 2:** designed for cleaner news and article text with lighter preprocessing.
- **Domain 3:** trained on a mixed sample from both domains to examine cross-domain generalization.

The project includes domain-aware sampling, heap-based BPE merge learning, entity-boundary markers, tokenizer serialization, performance testing, and a BiLSTM-based NER evaluation model.

## Technologies

Python, PyTorch, NumPy, regex, tqdm

## Project Files

- `base_tokenizer.py` - abstract tokenizer interface and serialization utilities
- `my_tokenizer.py` - BPE tokenizer for Domain 1
- `my_tokenizer_domain2.py` - BPE tokenizer for Domain 2
- `generate_tokenizers.py` - trains and saves all three tokenizers
- `train_tokenizer.py` - trains the Domain 1 tokenizer separately
- `test_tokenizer.py` - measures encoding speed, token efficiency, and reconstruction
- `train_ner_model.py` - trains and evaluates the BiLSTM NER model

## Installation

```bash
pip install torch numpy regex tqdm
```

## Usage

Train all three tokenizers:

```bash
python generate_tokenizers.py
```

Evaluate a trained tokenizer:

```bash
python test_tokenizer.py --tokenizer_path trained_tokenizers/tokenizer_1.pkl --train_file data/domain_1_train.txt --test_file data/domain_1_dev.txt
```

Train the NER model on Domain 1:

```bash
python train_ner_model.py --tokenizer_path trained_tokenizers/tokenizer_1.pkl --train_file data/ner_data/train_1_binary.tagged --dev_file data/ner_data/dev_1_binary.tagged
```

The dataset files are not included because of their size. Place them under `data/` using the paths shown in the commands above.

## Project Report

The [full project report](docs/project-report.pdf) describes the preprocessing, sampling strategy, BPE training process, and evaluation.
