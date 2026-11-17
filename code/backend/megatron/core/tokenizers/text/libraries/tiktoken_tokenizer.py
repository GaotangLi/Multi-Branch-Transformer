# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import base64
from copy import copy
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

try:
    import tiktoken
except ImportError:
    pass

from .abstract_tokenizer import MegatronTokenizerTextAbstract
from .chat_template import MegatronTokenizerChatTemplate

logger = logging.getLogger(__name__)

_PATTERN_TIKTOKEN_V1 = (
    r"[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
)
_PATTERN_TIKTOKEN_V2 = "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"  # pylint: disable=line-too-long
_PATTERN_LLAMA3 = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # pylint: disable=line-too-long
DEFAULT_TIKTOKEN_MAX_VOCAB = 2**17  # 131072
SPECIAL_TOKENS = ["<unk>", "<s>", "</s>", "<mask>", "<pad>", "<cls>", "<sep>"]
SPECIAL_TOKEN_TEMPLATE = "<SPECIAL_{id}>"
LLAMA3_SPECIAL_TOKENS = {
    "<|begin_of_text|>": 0,
    "<|end_of_text|>": 1,
    "<|fim_prefix|>": 2,
    "<|fim_middle|>": 3,
    "<|fim_end_fill|>": 253,
    "<|fim_pad|>": 254,
    "<|fim_suffix|>": 255,
}
TIKTOKEN_MAX_ENCODE_CHARS = 400_000

__all__ = ["TikTokenTokenizer", "reload_mergeable_ranks"]


def reload_mergeable_ranks(
    path: str, max_vocab: Optional[int] = None, num_special_tokens: Optional[int] = None
) -> Dict[bytes, int]:
    """
    Reload the tokenizer JSON file and convert it to Tiktoken format.

    Args:
        path (str): path to the tokenizer.
        max_vocab (Optional[int]): maximum size of vocabulary.
        num_special_tokens (Optional[int]): number of added special tokens.

    Returns:
        Dict[bytes, int]: reloaded tokenizer vocab.
    """

    assert path.endswith(".json")
    from megatron.core.utils import log_single_rank

    # reload vocab
    with open(path, "r") as f:
        vocab = json.load(f)
    assert isinstance(vocab, list)
    log_single_rank(logger, logging.INFO, f"Vocab size: {len(vocab)}")
    if max_vocab is not None:
        vocab = vocab[:max_vocab]
        from megatron.core.utils import log_single_rank

        log_single_rank(logger, logging.INFO, f"Cutting vocab to first {len(vocab)} tokens")

    # build ranks
    ranks: Dict[bytes, int] = {}
    for i, x in enumerate(vocab):
        assert x.keys() == {"rank", "token_bytes", "token_str"}
        assert x["rank"] == i
        merge = base64.b64decode(x["token_bytes"])
        assert i >= 256 or merge == bytes([i])
        ranks[merge] = x["rank"] + num_special_tokens

    # sanity check
    assert len(ranks) == len(vocab)
    assert set(ranks.values()) == set(range(num_special_tokens, len(ranks) + num_special_tokens))

    return ranks


class TikTokenTokenizer(MegatronTokenizerTextAbstract, MegatronTokenizerChatTemplate):
    """TikTokenTokenizer https://github.com/openai/tiktoken."""

    def __init__(
        self,
        tokenizer_path: str,
        special_tokens: Optional[List[str]] = None,
        num_special_tokens: Optional[int] = 1000,
        chat_template: Optional[str] = None,
        pattern: Optional[str] = "v2",
        vocab_size: Optional[int] = DEFAULT_TIKTOKEN_MAX_VOCAB,
    ):
        """
        Args:
            tokenizer_path (str): path to tokenizer vocabulary.
            special_tokens (Optional[List[str]]): template for user-defined special tokens.
            num_special_tokens (int): number of special tokens to generate.
            chat_template (Optional[str]): tokenizer chat template in jinja format.
            pattern (Optional[str]): regex pattern to split the text.
            vocab_size (Optional[int]): size of vocabulary.
        """

        if not tokenizer_path or not os.path.exists(tokenizer_path):
            raise ValueError(f"tokenizer_path: {tokenizer_path} is invalid")

        # Meta's Llama-3 ``original/tokenizer.model`` is a native tiktoken BPE
        # file, not Megatron's JSON-vocab format. It places the 128K merge
        # tokens first and 256 special tokens at the end. Treating it as the
        # generic tokenizer would silently permute every token ID.
        if Path(tokenizer_path).suffix == ".model":
            self._init_llama3(tokenizer_path, num_special_tokens, vocab_size)
            return

        if special_tokens is None:
            special_tokens = SPECIAL_TOKENS.copy()

        if pattern == "v1":
            pattern = _PATTERN_TIKTOKEN_V1
        elif pattern == "v2":
            pattern = _PATTERN_TIKTOKEN_V2
        else:
            raise ValueError(f"Expected tiktoken pattern to be `v1` or `v2`, but got {pattern}.")

        assert len(special_tokens) == len(
            set(special_tokens)
        ), f"Special tokens should be unique: {special_tokens}"
        assert len(special_tokens) <= num_special_tokens < vocab_size
        assert set(SPECIAL_TOKENS) <= set(
            special_tokens
        ), f"Custom special tokens should include {SPECIAL_TOKENS}"

        self._unk_id = special_tokens.index("<unk>")
        self._bos_id = special_tokens.index("<s>")
        self._eos_id = special_tokens.index("</s>")
        self._mask_id = special_tokens.index("<mask>")
        self._pad_id = special_tokens.index("<pad>")
        self._cls_id = special_tokens.index("<cls>")
        self._sep_id = special_tokens.index("<sep>")

        self._vocab_size = vocab_size
        self.chat_template = chat_template
        self.num_special_tokens = num_special_tokens
        special_filler = [
            SPECIAL_TOKEN_TEMPLATE.format(id=i)
            for i in range(len(special_tokens), num_special_tokens)
        ]
        self.special_filler = special_filler
        from megatron.core.utils import log_single_rank

        if special_filler:
            log_single_rank(
                logger,
                logging.INFO,
                "Adding special tokens: "
                f"{', '.join(special_tokens)}, {special_filler[0]}, ..., {special_filler[-1]}",
            )
        self.special_tokens = special_tokens + special_filler
        assert (
            len(set(self.special_tokens)) == len(self.special_tokens) == num_special_tokens
        ), self.special_tokens
        self.inner_vocab_size = vocab_size - num_special_tokens

        # reload vocab
        self.token2id = reload_mergeable_ranks(
            tokenizer_path, max_vocab=self.inner_vocab_size, num_special_tokens=num_special_tokens
        )

        self.id2token = {v: k for k, v in self.token2id.items()}
        assert set(range(num_special_tokens, vocab_size)) == set(self.id2token.keys())

        self.shifted_id2token = {i: tok for i, tok in enumerate(self.special_tokens)}
        for key, value in self.id2token.items():
            self.shifted_id2token[key + self.num_special_tokens] = value

        special_tokens_dict = {t: i for i, t in enumerate(self.special_tokens)}
        self.tokenizer = tiktoken.Encoding(
            name=Path(tokenizer_path).parent.name,
            pat_str=pattern,
            mergeable_ranks=self.token2id,
            special_tokens=special_tokens_dict,  # special tokens are handled manually
        )

        self._vocab = special_tokens_dict | self.token2id

    def _init_llama3(self, tokenizer_path: str, num_special_tokens: int, vocab_size: int):
        """Initialize Meta's native 128K-BPE + 256-special-token layout."""

        from tiktoken.load import load_tiktoken_bpe

        if num_special_tokens != 256:
            raise ValueError(
                "Llama-3 tokenizer.model requires --tiktoken-num-special-tokens 256"
            )
        mergeable_ranks = load_tiktoken_bpe(tokenizer_path)
        expected_vocab_size = len(mergeable_ranks) + num_special_tokens
        if vocab_size != expected_vocab_size:
            raise ValueError(
                f"Llama-3 tokenizer vocab mismatch: requested {vocab_size}, "
                f"expected {expected_vocab_size}"
            )

        relative_special_tokens = copy(LLAMA3_SPECIAL_TOKENS)
        missing_ids = set(range(num_special_tokens)) - set(relative_special_tokens.values())
        for token_id in sorted(missing_ids):
            relative_special_tokens[f"<|reserved_special_token_{token_id}|>"] = token_id
        special_tokens_dict = {
            token: len(mergeable_ranks) + token_id
            for token, token_id in relative_special_tokens.items()
        }

        self._llama3_layout = True
        self._vocab_size = vocab_size
        self.num_special_tokens = num_special_tokens
        self.inner_vocab_size = len(mergeable_ranks)
        self.special_tokens = [
            token
            for token, _ in sorted(special_tokens_dict.items(), key=lambda item: item[1])
        ]
        self.special_filler = [
            token for token in self.special_tokens if token.startswith("<|reserved_special_token_")
        ]
        self._special_token_to_id = special_tokens_dict
        self._special_ids = set(special_tokens_dict.values())
        self._bos_id = special_tokens_dict["<|begin_of_text|>"]
        self._eos_id = special_tokens_dict["<|end_of_text|>"]
        # GPT pretraining uses eod/pad. The other aliases exist because the
        # generic Megatron tokenizer interface requires them.
        self._unk_id = special_tokens_dict["<|reserved_special_token_4|>"]
        self._mask_id = special_tokens_dict["<|reserved_special_token_5|>"]
        self._pad_id = special_tokens_dict["<|fim_pad|>"]
        self._cls_id = self._bos_id
        self._sep_id = self._eos_id

        self.token2id = mergeable_ranks
        self.id2token = {token_id: token for token, token_id in mergeable_ranks.items()}
        self.shifted_id2token = self.id2token | {
            token_id: token for token, token_id in special_tokens_dict.items()
        }
        self.tokenizer = tiktoken.Encoding(
            name=Path(tokenizer_path).stem,
            pat_str=_PATTERN_LLAMA3,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens_dict,
        )
        self._vocab = mergeable_ranks | special_tokens_dict

    def text_to_tokens(self, text: str) -> List[str]:
        """Converts text to tokens."""
        token_ids = self.tokenizer.encode(text)
        return [self.tokenizer.decode_single_token_bytes(token) for token in token_ids]

    def tokens_to_text(self, tokens: List[int]) -> str:
        """Converts list of tokens to text."""
        token_ids = [self.tokenizer.encode_single_token(tokens) for tokens in tokens]
        return self.tokenizer.decode(token_ids)

    def token_to_id(self, token: str) -> int:
        """Converts a single token to it's id."""
        if getattr(self, "_llama3_layout", False):
            if token in self._special_token_to_id:
                return self._special_token_to_id[token]
            return self.tokenizer.encode_single_token(token)
        if token in self.special_tokens:
            return self.special_tokens.index(token)
        else:
            return self.tokenizer.encode_single_token(token) + self.num_special_tokens

    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Converts list of tokens to list of it's ids."""
        return [self.token_to_id(token) for token in tokens]

    def id_to_token(self, token_id: int) -> str:
        """Converts token id to token."""
        if getattr(self, "_llama3_layout", False):
            if token_id in self._special_ids:
                return self.shifted_id2token[token_id]
            token_bytes = self.tokenizer.decode_single_token_bytes(token_id)
            return token_bytes.decode('utf-8', errors='replace')
        if token_id < self.num_special_tokens:
            return self.special_tokens[token_id]
        else:
            token_bytes = self.tokenizer.decode_single_token_bytes(token_id)
            return token_bytes.decode('utf-8', errors='replace')

    def ids_to_tokens(self, token_ids: List[int]) -> List[str]:
        """Converts list of tokens ids to list of tokens."""
        tokens = []
        for token_id in token_ids:
            tokens.append(self.id_to_token(token_id))

        return tokens

    def text_to_ids(self, text: str) -> List[int]:
        """Converts text to list of ids."""
        if getattr(self, "_llama3_layout", False):
            # Ordinary encoding prevents a reserved-token spelling in corpus
            # text from being interpreted as control data.
            return sum(
                (
                    self.tokenizer.encode_ordinary(
                        text[start : start + TIKTOKEN_MAX_ENCODE_CHARS]
                    )
                    for start in range(0, len(text), TIKTOKEN_MAX_ENCODE_CHARS)
                ),
                start=[],
            )
        tokens = self.tokenizer.encode(text, allowed_special="all")
        return tokens

    def ids_to_text(self, tokens: List[int], remove_special_tokens: bool = False) -> str:
        """Converts list of ids to text."""
        # Filter out special tokens and adjust the remaining tokens
        if remove_special_tokens:
            if getattr(self, "_llama3_layout", False):
                adjusted_tokens = [t for t in tokens if t not in self._special_ids]
            else:
                adjusted_tokens = [
                    t
                    for t in tokens
                    if t not in {self.bos_id, self.eos_id} and t >= self.num_special_tokens
                ]
        else:
            adjusted_tokens = tokens

        # Use tiktoken's native decode which concatenates token bytes before
        # UTF-8 decoding, correctly handling multi-byte characters that span
        # multiple BPE tokens.
        if adjusted_tokens:
            return self.tokenizer.decode(adjusted_tokens)
        else:
            return ""  # Return an empty string if all tokens were filtered out

    def add_special_tokens(self, special_tokens_dict: dict):
        """Adds special tokens to the tokenizer."""
        raise NotImplementedError("This method is not supported for TikToken tokenizers.")

    def offsets(self, ids: list[int], text: str) -> list[int]:
        """Calculate offsets."""
        try:
            return self.tokenizer.decode_with_offsets(ids)[1]
        except UnicodeDecodeError:
            # Tiktoken has an unnecessary check that raises UnicodeDecodeError
            # from `text = b"".join(token_bytes).decode("utf-8", errors="strict")`
            # which is not needed for our use case. So we re-implement it, without
            # the check.

            token_bytes = self.tokenizer.decode_tokens_bytes(ids)
            text_len = 0
            offsets = []
            for token in token_bytes:
                offsets.append(max(0, text_len - (0x80 <= token[0] < 0xC0)))
                text_len += sum(1 for c in token if not 0x80 <= c < 0xC0)
            return offsets

    @property
    def additional_special_tokens_ids(self) -> list:
        """
        Returns a list of the additional special tokens, excluding [bos, eos, pad, unk]
        and special_filler. Used to return sentinel tokens for e.g. T5.
        """
        excluding_tokens = (
            self.ids_to_tokens([self._unk_id, self._bos_id, self._eos_id]) + self.special_filler
        )
        result = [
            self.token_to_id(token)
            for token in self.special_tokens
            if token not in excluding_tokens
        ]
        return result

    @property
    def bos_id(self) -> int:
        """Returns id of beginning of sentence token."""
        return self._bos_id

    @property
    def eos_id(self) -> int:
        """Returns id of end of sentence token."""
        return self._eos_id

    @property
    def eod(self) -> int:
        """Returns id of end of document token."""
        return self._eos_id

    @property
    def unk_id(self) -> int:
        """Returns id of unknown tokens."""
        return self._unk_id

    @property
    def mask_id(self) -> int:
        """Returns id of mask token."""
        return self._mask_id

    @property
    def pad_id(self) -> int:
        """Returns id of padding token."""
        return self._pad_id

    @property
    def cls_id(self) -> int:
        """Returns id of classification token."""
        return self._cls_id

    @property
    def sep_id(self) -> int:
        """Returns id of SEP token."""
        return self._sep_id

    @property
    def vocab(self):
        """Returns tokenizer vocab."""
        return self._vocab

    @property
    def decoder(self):
        """ """
        return self.shifted_id2token

    @property
    def encoder(self):
        """ """
        return self._vocab

    @property
    def vocab_size(self) -> int:
        """Returns tokenizer vocab size."""
        return self._vocab_size

    @property
    def inv_vocab(self) -> dict:
        """Returns tokenizer vocab with reversed keys and values."""
        return self.shifted_id2token
