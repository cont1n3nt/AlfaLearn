"""Local multilingual E5 encoding and passage scoring."""

import os

import numpy as np

MODEL_ID = "intfloat/multilingual-e5-base"
MODEL_REVISION = "d13f1b27baf31030b7fd040960d60d909913633f"
MODEL_FILE = "onnx/model.onnx"
MODEL_LIMIT = 512


class E5Encoder:
    def __init__(self, tokenizer_path, model_path, *, truncate=False):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.no_truncation()
        if truncate:
            self.tokenizer.enable_truncation(max_length=MODEL_LIMIT)
        self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id("<pad>"), pad_token="<pad>")
        options = ort.SessionOptions()
        options.intra_op_num_threads = min(os.cpu_count() or 1, 4)
        self.session = ort.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )

    def encode(self, texts):
        encoded = self.tokenizer.encode_batch(texts, add_special_tokens=True)
        if any(sum(row.attention_mask) > MODEL_LIMIT for row in encoded):
            raise ValueError(f"вход E5 длиннее {MODEL_LIMIT} токенов")
        inputs = {
            "input_ids": np.asarray([row.ids for row in encoded], dtype=np.int64),
            "attention_mask": np.asarray([row.attention_mask for row in encoded], dtype=np.int64),
        }
        hidden = self.session.run(["last_hidden_state"], inputs)[0]
        mask = inputs["attention_mask"][..., None].astype(np.float32)
        pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        if not np.isfinite(norms).all() or np.any(norms == 0):
            raise ValueError("E5 вернула некорректные эмбеддинги")
        return (pooled / norms).astype(np.float32)


class SemanticIndex:
    def __init__(self, chunks, model_dir):
        tokenizer_path, model_path = model_dir / "tokenizer.json", model_dir / MODEL_FILE
        if not tokenizer_path.is_file() or not model_path.is_file():
            raise ValueError("В AI_E5_MODEL_DIR нужны tokenizer.json и onnx/model.onnx.")
        self.encoder = E5Encoder(tokenizer_path, model_path, truncate=True)
        texts = [
            f"passage: {chunk['text']}"
            if chunk["document_id"].startswith("podcast:")
            else chunk["embedding_text"]
            for chunk in chunks
        ]
        self.vectors = np.concatenate(
            [self.encoder.encode(texts[i : i + 16]) for i in range(0, len(texts), 16)]
        )

    def score(self, queries):
        return self.encoder.encode([f"query: {question}" for question in queries]) @ self.vectors.T
