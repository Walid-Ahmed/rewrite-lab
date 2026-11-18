"""Shared data, model, metric and comparison routines for the fresh-encoder fine-tuning comparison."""
import os
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
import hashlib
import json
import sys
from pathlib import Path

P = Path(__file__).resolve().parent
ROOT = P.parent / 'bert'  # train_score_bert.ScoreBert
H = Path(os.environ.get('DATA_ROOT', 'data')).resolve()  # private data and model folders, not in this repo
sys.path.insert(0, str(P / 'vendor'))
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from train_score_bert import ScoreBert

DATASET = H / 'model2-retrain-balanced-20261029/dataset-expanded/model2-detector-a'
EXISTING = H / 'model2-retrain-balanced-20261029/model-expanded/last-model.pt'
EXISTING_TOKENIZER = H / 'model2-retrain-balanced-20261029/model-expanded/tokenizer'
HOLDOUT = H / 'model2-high-score-audit-20261101/holdout-manifest.json'
ROW_METADATA = H / 'model2-feature-pilot-20261101/row-metadata.json'
ENCODERS = {
    'bert': {'model': 'google-bert/bert-base-uncased', 'revision': '86b5e0934494bd15c9632b12f734a8a67f723594'},
    'deberta': {'model': 'microsoft/deberta-v3-base', 'revision': '8ccc9b6f36199bec6961081d44eb72fb3f7353f3'},
}

def read(path):
    return json.loads(Path(path).read_text())

def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def band(score):
    return 'zero' if score == 0 else 'hundred' if score == 100 else 'intermediate'

class ScoreDeberta(nn.Module):
    """Same head shape as ScoreBert: [CLS] -> dense+tanh pooler -> dropout -> linear -> sigmoid.

    DeBERTa-v3 ships no pretrained pooler, so the dense+tanh layer starts fresh, unlike BERT's.
    """
    def __init__(self, model_name, revision=None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, revision=revision, dtype=torch.float32)
        hidden = self.encoder.config.hidden_size
        self.pooler = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.head = nn.Sequential(nn.Dropout(.1), nn.Linear(hidden, 1))
    def forward(self, **inputs):
        cls = self.encoder(**inputs).last_hidden_state[:, 0]
        return torch.sigmoid(self.head(self.pooler(cls))).squeeze(-1)

class SentencePieceTokenizer:
    """Reference DeBERTa-v3 tokenization straight from spm.model.

    The Transformers 5 conversion drops SentencePiece's NFKC/control-character normalization and byte
    fallback, turning µ, ε, superscripts and similar into [UNK] (868 tokens across 221 corpus texts).
    """
    def __init__(self, model_name, revision):
        import sentencepiece
        from huggingface_hub import hf_hub_download
        self.path = Path(hf_hub_download(model_name, 'spm.model', revision=revision))
        self.sp = sentencepiece.SentencePieceProcessor(model_file=str(self.path))
        ids = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.cls_token_id, self.sep_token_id, self.pad_token_id = ids.cls_token_id, ids.sep_token_id, ids.pad_token_id
    def encode_one(self, text, truncation, max_length):
        pieces = self.sp.encode(text)
        if truncation and max_length:
            pieces = pieces[:max_length - 2]
        ids = [self.cls_token_id, *pieces, self.sep_token_id]
        return {'input_ids': ids, 'attention_mask': [1] * len(ids)}
    def __call__(self, text, truncation=False, max_length=None, padding=False, return_tensors=None):
        if isinstance(text, str):
            assert not padding and return_tensors is None
            return self.encode_one(text, truncation, max_length)
        return self.pad([self.encode_one(t, truncation, max_length) for t in text], return_tensors=return_tensors)
    def pad(self, rows, return_tensors='pt'):
        width = max(len(r['input_ids']) for r in rows)
        ids = [r['input_ids'] + [self.pad_token_id] * (width - len(r['input_ids'])) for r in rows]
        mask = [r['attention_mask'] + [0] * (width - len(r['attention_mask'])) for r in rows]
        assert return_tensors == 'pt'
        return {'input_ids': torch.tensor(ids), 'attention_mask': torch.tensor(mask)}
    def save_pretrained(self, directory):
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        (directory / 'spm.model').write_bytes(self.path.read_bytes())
        write(directory / 'special-tokens.json', {'cls': self.cls_token_id, 'sep': self.sep_token_id, 'pad': self.pad_token_id,
                                                  'tokenizer': 'sentencepiece reference, see common.SentencePieceTokenizer'})

def build(encoder):
    spec = ENCODERS[encoder]
    if encoder == 'bert':
        # The repo's production architecture, unchanged, so predict_score_bert.py can load the result.
        model = ScoreBert(spec['model'])
        tokenizer = AutoTokenizer.from_pretrained(spec['model'], revision=spec['revision'])
    else:
        model = ScoreDeberta(spec['model'], spec['revision'])
        tokenizer = SentencePieceTokenizer(spec['model'], spec['revision'])
    # Both arms recompute activations in backward so they fit beside other GPU users; the math is unchanged.
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    return model, tokenizer

def load_existing():
    saved = torch.load(EXISTING, map_location='cpu', weights_only=False)
    model = ScoreBert(saved['config']['model']); model.load_state_dict(saved['model'])
    return model, AutoTokenizer.from_pretrained(EXISTING_TOKENIZER)

def splits():
    from training_data import load_training_data
    return load_training_data('model2', DATASET)

def predict(model, tokenizer, rows, device, max_length=512, batch_size=32):
    model.eval().to(device); out = []
    with torch.inference_mode():
        for i in range(0, len(rows), batch_size):
            batch = tokenizer([r['text'] for r in rows[i:i + batch_size]], truncation=True,
                              max_length=max_length, padding=True, return_tensors='pt')
            out.extend((model(**{k: v.to(device) for k, v in batch.items()}) * 100).cpu().tolist())
    return np.array(out, dtype=np.float64)

def metrics(prediction, gold):
    gold = np.asarray(gold); error = np.asarray(prediction) - gold
    def mae(mask):
        return float(np.mean(abs(error[mask]))) if mask.sum() else None
    return {'mae_pp': float(np.mean(abs(error))), 'rmse_pp': float(np.sqrt(np.mean(error**2))),
            'hundred_mae_pp': mae(gold == 100), 'hundred_rmse_pp': float(np.sqrt(np.mean(error[gold == 100]**2))),
            'zero_mae_pp': mae(gold == 0), 'intermediate_mae_pp': mae((gold > 0) & (gold < 100))}

def gate(candidate, comparator):
    checks = {'lower_hundred_mae': candidate['hundred_mae_pp'] < comparator['hundred_mae_pp'],
              'non_worse_overall_mae': candidate['mae_pp'] <= comparator['mae_pp'],
              'non_worse_overall_rmse': candidate['rmse_pp'] <= comparator['rmse_pp']}
    return {'checks': checks, 'passed': all(checks.values())}

def clustered_contrast(candidate, comparator, gold, groups, seed, replicates):
    """Paired whole-family bootstrap of candidate minus comparator; negative favours the candidate."""
    gold = np.asarray(gold); _, inverse = np.unique(groups, return_inverse=True)
    count = np.bincount(inverse); high_count = np.bincount(inverse, weights=gold == 100)
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(count), size=(replicates, len(count)))
    n, nh = count[ix].sum(axis=1), high_count[ix].sum(axis=1)
    assert (nh > 0).all()
    ea, eb = np.asarray(candidate) - gold, np.asarray(comparator) - gold
    sums = lambda a: np.bincount(inverse, weights=a, minlength=len(count))[ix].sum(axis=1)
    bootstrap = {'mae_pp': sums(abs(ea) - abs(eb)) / n,
                 'hundred_mae_pp': sums((abs(ea) - abs(eb)) * (gold == 100)) / nh,
                 'rmse_pp': np.sqrt(sums(ea**2) / n) - np.sqrt(sums(eb**2) / n)}
    a, b = metrics(candidate, gold), metrics(comparator, gold)
    return {name: {'difference_pp': a[name] - b[name],
                   'family_cluster_95pct_interval': np.quantile(values, [.025, .975]).tolist()}
            for name, values in bootstrap.items()}
