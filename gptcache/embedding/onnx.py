import numpy as np

from gptcache.embedding.base import BaseEmbedding
from gptcache.utils import (
    import_onnxruntime,
    import_huggingface_hub,
    import_huggingface,
)

import_huggingface()
import_onnxruntime()
import_huggingface_hub()

from transformers import AutoTokenizer, AutoConfig  # pylint: disable=C0413
from huggingface_hub import hf_hub_download  # pylint: disable=C0413
import onnxruntime  # pylint: disable=C0413

class Onnx(BaseEmbedding):
    """Generate text embedding for given text using ONNX Model.

    Example:
        .. code-block:: python

            from gptcache.embedding import Onnx

            test_sentence = 'Hello, world.'
            encoder = Onnx(model='GPTCache/paraphrase-albert-onnx')
            embed = encoder.to_embeddings(test_sentence)
    """

    def __init__(self, model="GPTCache/paraphrase-albert-onnx", chunk_overlap=64):
        tokenizer_name = "GPTCache/paraphrase-albert-small-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = model
        onnx_model_path = hf_hub_download(repo_id=model, filename="model.onnx")
        self.ort_session = onnxruntime.InferenceSession(onnx_model_path)
        config = AutoConfig.from_pretrained(
            "GPTCache/paraphrase-albert-small-v2"
        )

        self.__dimension = config.hidden_size
        tokenizer_limit = getattr(self.tokenizer, "model_max_length", 512)
        config_limit = getattr(config, "max_position_embeddings", None)
        self.max_length = config_limit or tokenizer_limit or 512
        if self.max_length is None or self.max_length > 100000:
            self.max_length = 512
        special_tokens = self.tokenizer.num_special_tokens_to_add(pair=False)
        max_chunk_overlap = max(0, self.max_length - special_tokens - 1)
        self.chunk_overlap = max(0, min(chunk_overlap, max_chunk_overlap))
        self.input_names = {input_meta.name for input_meta in self.ort_session.get_inputs()}

    def to_embeddings(self, data, **_):
        """Generate embedding given text input.

        :param data: text in string.
        :type data: str

        :return: a text embedding in shape of (dim,).
        """
        encoded_text = self.tokenizer(
            data,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            stride=self.chunk_overlap,
            return_overflowing_tokens=True,
            return_tensors="np",
        )

        attention_mask = encoded_text.get("attention_mask")
        if attention_mask is None and "input_ids" in encoded_text:
            attention_mask = (encoded_text["input_ids"] != self.tokenizer.pad_token_id).astype("int64")
        attention_mask = attention_mask.astype("int64")

        chunk_embeddings = []
        chunk_weights = []
        num_chunks = attention_mask.shape[0]
        for chunk_idx in range(num_chunks):
            ort_inputs = {}
            if "input_ids" in self.input_names:
                ort_inputs["input_ids"] = encoded_text["input_ids"][chunk_idx : chunk_idx + 1].astype("int64")
            if "attention_mask" in self.input_names:
                ort_inputs["attention_mask"] = attention_mask[chunk_idx : chunk_idx + 1]
            if "token_type_ids" in self.input_names and "token_type_ids" in encoded_text:
                ort_inputs["token_type_ids"] = encoded_text["token_type_ids"][chunk_idx : chunk_idx + 1].astype("int64")

            ort_outputs = self.ort_session.run(None, ort_inputs)
            ort_feat = ort_outputs[0]
            chunk_emb = self.post_proc(ort_feat, attention_mask[chunk_idx : chunk_idx + 1]).flatten()
            chunk_embeddings.append(chunk_emb)
            chunk_weights.append(max(int(attention_mask[chunk_idx].sum()), 1))

        emb = np.vstack(chunk_embeddings)
        if emb.shape[0] == 1:
            return emb[0]

        # Pool chunk embeddings back into one vector so long prompts can exceed
        # the encoder window while keeping a fixed-size embedding output.
        chunk_weights = np.array(chunk_weights, dtype=float).reshape(-1, 1)
        pooled_emb = np.sum(emb * chunk_weights, axis=0) / np.sum(chunk_weights)
        return pooled_emb.flatten()
    
    def post_proc(self, token_embeddings, attention_mask):
        input_mask_expanded = (
            np.expand_dims(attention_mask, -1)
            .repeat(token_embeddings.shape[-1], -1)
            .astype(float)
        )
        sentence_embs = np.sum(token_embeddings * input_mask_expanded, 1) / np.maximum(
            input_mask_expanded.sum(1), 1e-9
        )
        return sentence_embs

    @property
    def dimension(self):
        """Embedding dimension.

        :return: embedding dimension
        """
        return self.__dimension
