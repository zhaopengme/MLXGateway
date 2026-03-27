"""GGUF model loading support for MLXGateway.

Supports loading GGUF-format LLMs (Llama, Qwen, Phi, Gemma, etc.) directly into
the MLX inference pipeline, compatible with the existing ModelCache and ChatGenerator.

Supported model_id formats:
  - Local file:           /path/to/model.gguf
  - HF repo (auto-pick): unsloth/Qwen3-8B-GGUF
  - HF repo + quantize:  unsloth/Qwen3-8B-GGUF:Q4_K_M   (matches *Q4_K_M*.gguf)
  - HF repo + filename:  unsloth/Qwen3-8B-GGUF:qwen3-8b-q4_k_m.gguf
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..utils.logger import logger


# ---------------------------------------------------------------------------
# Quantization type constants (matches GGUF spec file_type field)
# ---------------------------------------------------------------------------
_GGUF_FTYPE_F32 = 0
_GGUF_FTYPE_F16 = 1
_GGUF_FTYPE_Q4_0 = 2
_GGUF_FTYPE_Q4_1 = 3
_GGUF_FTYPE_Q8_0 = 7
_GGUF_FTYPE_Q5_0 = 8
_GGUF_FTYPE_Q5_1 = 9
_GGUF_FTYPE_Q2_K = 10
_GGUF_FTYPE_Q3_K_S = 11
_GGUF_FTYPE_Q3_K_M = 12
_GGUF_FTYPE_Q3_K_L = 13
_GGUF_FTYPE_Q4_K_S = 14
_GGUF_FTYPE_Q4_K_M = 15
_GGUF_FTYPE_Q5_K_S = 16
_GGUF_FTYPE_Q5_K_M = 17
_GGUF_FTYPE_Q6_K = 18
_GGUF_FTYPE_Q8_K = 19
_GGUF_FTYPE_BF16 = 32

# Quantization families: maps ftype -> (bits, group_size) for nn.quantize
_QUANTIZATION_MAP: Dict[int, Tuple[int, int]] = {
    _GGUF_FTYPE_Q4_0: (4, 32),
    _GGUF_FTYPE_Q4_1: (4, 32),
    _GGUF_FTYPE_Q4_K_S: (4, 32),
    _GGUF_FTYPE_Q4_K_M: (4, 32),
    _GGUF_FTYPE_Q8_0: (8, 32),
    _GGUF_FTYPE_Q8_K: (8, 32),
    _GGUF_FTYPE_Q5_0: (4, 32),   # round down to 4-bit for MLX
    _GGUF_FTYPE_Q5_1: (4, 32),
    _GGUF_FTYPE_Q5_K_S: (4, 32),
    _GGUF_FTYPE_Q5_K_M: (4, 32),
    _GGUF_FTYPE_Q6_K: (8, 32),   # round up to 8-bit for MLX
    _GGUF_FTYPE_Q2_K: (4, 32),
    _GGUF_FTYPE_Q3_K_S: (4, 32),
    _GGUF_FTYPE_Q3_K_M: (4, 32),
    _GGUF_FTYPE_Q3_K_L: (4, 32),
}

# Preferred quantization files (lower number = higher preference)
_QUANT_PREFERENCE = [
    "Q4_K_M", "Q5_K_M", "Q4_K_S", "Q5_K_S",
    "Q4_0", "Q8_0", "Q6_K", "Q3_K_M", "Q2_K",
]


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

@dataclass
class GGUFModelArgs:
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    context_length: int
    num_key_value_heads: Optional[int] = None
    rope_theta: float = 10000.0
    rope_traditional: bool = False
    head_dim: Optional[int] = None

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads


# ---------------------------------------------------------------------------
# MLX model definition (Llama / Qwen architecture)
# ---------------------------------------------------------------------------

class _Attention(nn.Module):
    def __init__(self, args: GGUFModelArgs):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim ** -0.5

        dim = args.hidden_size
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.rope = nn.RoPE(
            self.head_dim,
            traditional=args.rope_traditional,
            base=args.rope_theta,
        )

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = q.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        if cache is not None:
            key_cache, val_cache = cache
            q = self.rope(q, offset=key_cache.shape[2])
            k = self.rope(k, offset=key_cache.shape[2])
            k = mx.concatenate([key_cache, k], axis=2)
            v = mx.concatenate([val_cache, v], axis=2)
        else:
            q = self.rope(q)
            k = self.rope(k)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out), (k, v)


class _MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class _TransformerBlock(nn.Module):
    def __init__(self, args: GGUFModelArgs):
        super().__init__()
        self.self_attn = _Attention(args)
        self.mlp = _MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, x, mask=None, cache=None):
        r, cache = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r, cache


class _LlamaModel(nn.Module):
    def __init__(self, args: GGUFModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [_TransformerBlock(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs, cache=None):
        h = self.embed_tokens(inputs)
        mask = None
        if h.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
            mask = mask.astype(h.dtype)
        if cache is None:
            cache = [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            h, cache[i] = layer(h, mask, cache[i])
        return self.norm(h), cache


class _GGUFConfigShim:
    """Shim that exposes GGUFModelArgs fields under the names mlx-lm expects.

    mlx_lm.models.cache.make_prompt_cache reads model.args (or model.config)
    for n_kv_heads, head_dim, etc.  This shim maps our GGUFModelArgs fields
    to the attribute names mlx-lm uses so prompt caching works transparently.
    """

    def __init__(self, args: GGUFModelArgs):
        self._args = args

    def __getattr__(self, name: str):
        # Direct field access on the underlying dataclass
        a = object.__getattribute__(self, "_args")
        if hasattr(a, name):
            return getattr(a, name)

        # mlx-lm compat aliases
        _ALIASES = {
            "n_kv_heads": "num_key_value_heads",
            "num_layers": "num_hidden_layers",
            "max_position_embeddings": "context_length",
            "n_positions": "context_length",
            "max_sequence_length": "context_length",
            "model_max_length": "context_length",
        }
        target = _ALIASES.get(name)
        if target and hasattr(a, target):
            return getattr(a, target)

        raise AttributeError(f"_GGUFConfigShim has no attribute '{name}'")


class GGUFLanguageModel(nn.Module):
    """MLX language model loaded from a GGUF file. Compatible with mlx-lm ChatGenerator."""

    def __init__(self, args: GGUFModelArgs):
        super().__init__()
        self.args = _GGUFConfigShim(args)
        self.model = _LlamaModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None):
        out, cache = self.model(inputs, cache)
        return self.lm_head(out), cache

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.head_dim

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads


# ---------------------------------------------------------------------------
# Tokenizer built from GGUF metadata
# ---------------------------------------------------------------------------

class GGUFTokenizer:
    """Minimal tokenizer extracted from GGUF metadata (sentencepiece-based).

    Wraps the sentencepiece tokenizer stored inside GGUF metadata, or falls back
    to the HuggingFace tokenizer from the same repo when SPM data is absent.
    """

    def __init__(self, metadata: Dict[str, Any], repo_id: Optional[str] = None):
        self._spm = None
        self._hf_tokenizer = None
        self.eos_token_id: int = 2
        self.bos_token_id: int = 1

        # Try sentencepiece data embedded in metadata
        if "tokenizer.ggml.tokens" in metadata:
            try:
                self._spm = _build_spm_tokenizer(metadata)
                self.eos_token_id = int(_mk(metadata.get("tokenizer.ggml.eos_token_id", 2)))
                self.bos_token_id = int(_mk(metadata.get("tokenizer.ggml.bos_token_id", 1)))
                logger.info("GGUF tokenizer: using embedded sentencepiece")
                return
            except Exception as e:
                logger.warning(f"GGUF SPM tokenizer failed ({e}), trying HF fallback")

        # Fall back to HuggingFace tokenizer from the same repo
        if repo_id:
            try:
                from transformers import AutoTokenizer
                self._hf_tokenizer = AutoTokenizer.from_pretrained(
                    repo_id, trust_remote_code=True
                )
                self.eos_token_id = self._hf_tokenizer.eos_token_id or 2
                self.bos_token_id = self._hf_tokenizer.bos_token_id or 1
                logger.info(f"GGUF tokenizer: using HuggingFace tokenizer from {repo_id}")
            except Exception as e:
                raise RuntimeError(
                    f"Could not construct tokenizer from GGUF metadata or HF repo '{repo_id}': {e}"
                )
        else:
            raise RuntimeError(
                "GGUF file has no embedded tokenizer and no repo_id provided for HF fallback."
            )

    def encode(self, text: str) -> List[int]:
        if self._spm is not None:
            return [self.bos_token_id] + self._spm.encode(text)
        return self._hf_tokenizer.encode(text)

    def decode(self, tokens: List[int]) -> str:
        if self._spm is not None:
            return self._spm.decode(tokens)
        return self._hf_tokenizer.decode(tokens)

    # mlx-lm compatibility shim -----------------------------------------------

    def apply_chat_template(self, messages, tokenize: bool = True, add_generation_prompt: bool = True, **kwargs):
        """Delegate to HF tokenizer when available; otherwise build a simple template."""
        if self._hf_tokenizer is not None:
            return self._hf_tokenizer.apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
                **kwargs,
            )
        # Fallback simple template (Llama-style)
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|{role}|>\n{content}<|end|>\n")
        if add_generation_prompt:
            parts.append("<|assistant|>\n")
        text = "".join(parts)
        if tokenize:
            return self.encode(text)
        return text

    @property
    def model_max_length(self) -> int:
        if self._hf_tokenizer is not None:
            return getattr(self._hf_tokenizer, "model_max_length", 4096)
        return 4096

    def __getattr__(self, name: str):
        """Proxy unknown attribute accesses to the HF tokenizer when available."""
        if name.startswith("_"):
            raise AttributeError(name)
        # Use object.__getattribute__ to avoid recursion if _hf_tokenizer
        # hasn't been set yet (e.g. exception during __init__).
        try:
            hf = object.__getattribute__(self, "_hf_tokenizer")
        except AttributeError:
            raise AttributeError(f"GGUFTokenizer has no attribute '{name}'")
        if hf is not None:
            return getattr(hf, name)
        raise AttributeError(f"GGUFTokenizer has no attribute '{name}'")


def _build_spm_tokenizer(metadata: Dict[str, Any]):
    """Build a sentencepiece tokenizer from GGUF metadata."""
    try:
        import sentencepiece as spm
    except ImportError:
        raise ImportError("Install sentencepiece: pip install sentencepiece")

    model_data = metadata.get("tokenizer.ggml.model")
    if model_data is None:
        raise ValueError("No sentencepiece model found in GGUF metadata")

    # model_data may be an mx.array or bytes
    if hasattr(model_data, "tobytes"):
        model_data = model_data.tobytes()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".model") as f:
        f.write(model_data)
        tmp_path = f.name
    try:
        sp = spm.SentencePieceProcessor()
        sp.Load(tmp_path)
        return sp
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Architecture detection & weight name translation
# ---------------------------------------------------------------------------

def detect_architecture(metadata: Dict[str, Any]) -> str:
    """Return a normalized architecture name from GGUF metadata."""
    arch = metadata.get("general.architecture", "")
    if hasattr(arch, "item"):
        arch = arch.item()
    return str(arch).lower()


def _mk(v):
    """Unwrap mx.array scalar to Python value."""
    if hasattr(v, "item"):
        return v.item()
    return v


def extract_model_config(metadata: Dict[str, Any]) -> GGUFModelArgs:
    """Extract GGUFModelArgs from GGUF metadata for any supported architecture."""
    arch = detect_architecture(metadata)

    def key(*suffixes):
        for s in suffixes:
            full = f"{arch}.{s}"
            if full in metadata:
                return _mk(metadata[full])
        raise KeyError(f"None of {[arch + '.' + s for s in suffixes]} found in GGUF metadata")

    hidden_size = key("embedding_length")
    num_layers = key("block_count")
    num_heads = key("attention.head_count")
    intermediate_size = key("feed_forward_length")
    norm_eps = key("attention.layer_norm_rms_epsilon", "attention.layer_norm_epsilon")
    context_length = key("context_length")

    # Vocab size: prefer the tokenizer token list length (most reliable), then
    # try ggml-specific vocab key, and finally fall back to an arch-specific key.
    if "tokenizer.ggml.tokens" in metadata:
        vocab_size = len(_mk(metadata["tokenizer.ggml.tokens"]))
    elif "tokenizer.ggml.vocab_size" in metadata:
        vocab_size = int(_mk(metadata["tokenizer.ggml.vocab_size"]))
    else:
        vocab_size = key("vocab_size")

    try:
        kv_heads = _mk(metadata.get(f"{arch}.attention.head_count_kv", num_heads))
    except Exception:
        kv_heads = num_heads

    try:
        rope_theta = _mk(metadata.get(f"{arch}.rope.freq_base", 10000.0))
    except Exception:
        rope_theta = 10000.0

    return GGUFModelArgs(
        hidden_size=int(hidden_size),
        num_hidden_layers=int(num_layers),
        intermediate_size=int(intermediate_size),
        num_attention_heads=int(num_heads),
        num_key_value_heads=int(kv_heads),
        rms_norm_eps=float(norm_eps),
        vocab_size=int(vocab_size),
        context_length=int(context_length),
        rope_theta=float(rope_theta),
    )


_WEIGHT_REGEX_RULES = [
    (re.compile(r"^blk\.(\d+)\."), lambda m: f"model.layers.{m.group(1)}."),
]

# Ordered from most-specific to least-specific to avoid partial matches.
# "output_norm" and "attn_output" must be matched before bare "output".
_WEIGHT_STRING_RULES = [
    ("attn_output.weight", "self_attn.o_proj.weight"),
    ("attn_q.weight", "self_attn.q_proj.weight"),
    ("attn_k.weight", "self_attn.k_proj.weight"),
    ("attn_v.weight", "self_attn.v_proj.weight"),
    ("attn_norm.weight", "input_layernorm.weight"),
    ("ffn_gate.weight", "mlp.gate_proj.weight"),
    ("ffn_down.weight", "mlp.down_proj.weight"),
    ("ffn_up.weight", "mlp.up_proj.weight"),
    ("ffn_norm.weight", "post_attention_layernorm.weight"),
    ("token_embd.weight", "model.embed_tokens.weight"),
    ("output_norm.weight", "model.norm.weight"),
    ("output.weight", "lm_head.weight"),
]


def translate_weight_names(name: str) -> str:
    """Map GGUF tensor names to the MLX Llama model attribute path."""
    result = name

    for pattern, replacement in _WEIGHT_REGEX_RULES:
        result = pattern.sub(replacement, result)

    # Apply at most one string rule per name (first match wins).
    for old, new in _WEIGHT_STRING_RULES:
        if old in result:
            result = result.replace(old, new)
            break

    return result


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------

def _detect_quantization(metadata: Dict[str, Any], weights: Dict[str, mx.array]):
    """Return (bits, group_size) or None if the model is not quantized."""
    ftype = metadata.get("general.file_type")
    if ftype is not None:
        ftype = int(_mk(ftype))
        if ftype in _QUANTIZATION_MAP:
            return _QUANTIZATION_MAP[ftype]
        if ftype in (_GGUF_FTYPE_F32, _GGUF_FTYPE_F16, _GGUF_FTYPE_BF16):
            return None
        logger.warning(f"Unknown GGUF file_type {ftype}, assuming no quantization")
        return None

    # Heuristic: if any weight has a "scales" sibling, treat as 4-bit
    if any(k.endswith(".scales") for k in weights):
        return (4, 32)
    return None


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model_from_gguf(
    weights: Dict[str, mx.array],
    metadata: Dict[str, Any],
) -> GGUFLanguageModel:
    """Construct and return a GGUFLanguageModel with weights loaded."""
    args = extract_model_config(metadata)
    quant = _detect_quantization(metadata, weights)

    translated = {translate_weight_names(k): v for k, v in weights.items()}

    model = GGUFLanguageModel(args)

    if quant is not None:
        bits, group_size = quant

        def class_predicate(path: str, module) -> bool:
            scales_key = f"{path}.scales"
            return (
                isinstance(module, (nn.Linear, nn.Embedding))
                and scales_key in translated
            )

        nn.quantize(model, bits=bits, group_size=group_size, class_predicate=class_predicate)
        logger.info(f"GGUF model quantized: bits={bits} group_size={group_size}")

    model.load_weights(list(translated.items()), strict=False)
    mx.eval(model.parameters())
    logger.info(
        f"GGUF model built: layers={args.num_hidden_layers} "
        f"hidden={args.hidden_size} heads={args.num_attention_heads} "
        f"kv_heads={args.num_key_value_heads} ctx={args.context_length}"
    )
    return model


# ---------------------------------------------------------------------------
# Public API: model_id resolution
# ---------------------------------------------------------------------------

_SUPPORTED_ARCHITECTURES = frozenset({
    "llama", "qwen2", "qwen3", "phi", "phi3", "gemma", "gemma2", "mistral"
})


def is_gguf_model(model_id: str) -> bool:
    """Return True if model_id points to a GGUF file or a known GGUF HF repo.

    Recognized formats:
      - /absolute/path/to/model.gguf
      - relative/path.gguf
      - owner/repo-GGUF                  (repo name ends with -GGUF or _GGUF)
      - owner/repo:Q4_K_M                (quantization suffix)
      - owner/repo:filename.gguf         (explicit .gguf filename)
    """
    s = model_id.strip()
    # Local file
    if s.lower().endswith(".gguf"):
        return True
    # HF repo with explicit quantization or file suffix
    if ":" in s:
        _, suffix = s.split(":", 1)
        if suffix.lower().endswith(".gguf"):
            return True
        # Quantization tag like Q4_K_M, q8_0, etc.
        if re.match(r"^[Qq]\d", suffix):
            return True
    # HF repo whose name ends with -gguf or _gguf
    repo_part = s.split(":")[0]
    if re.search(r"[-_]gguf$", repo_part, re.IGNORECASE):
        return True
    return False


def resolve_gguf_file(model_id: str) -> Tuple[str, Optional[str]]:
    """Resolve a model_id to (local_gguf_path, hf_repo_id).

    Downloads from HuggingFace if necessary.
    Returns (path_to_gguf_file, repo_id_or_None).
    """
    s = model_id.strip()

    # Local file path: ends with .gguf and has no ":" separator
    if s.lower().endswith(".gguf") and ":" not in s:
        p = Path(s)
        if not p.exists():
            raise FileNotFoundError(f"GGUF file not found: {s}")
        return str(p), None

    # HF repo (with optional :quantization or :filename suffix)
    repo_id, file_hint = (s.split(":", 1) + [None])[:2]

    gguf_filename = _resolve_gguf_filename(repo_id, file_hint)
    local_path = _download_gguf(repo_id, gguf_filename)
    return local_path, repo_id


def _resolve_gguf_filename(repo_id: str, hint: Optional[str]) -> str:
    """Pick a GGUF filename from the HF repo, given an optional hint."""
    from huggingface_hub import list_repo_files

    all_files = [f for f in list_repo_files(repo_id) if f.lower().endswith(".gguf")]
    if not all_files:
        raise FileNotFoundError(f"No .gguf files found in HuggingFace repo '{repo_id}'")

    if hint:
        # Exact filename match
        if hint.lower().endswith(".gguf"):
            matches = [f for f in all_files if f.lower() == hint.lower() or Path(f).name.lower() == hint.lower()]
            if matches:
                return matches[0]
        # Quantization tag match (e.g. Q4_K_M -> look for *Q4_K_M*.gguf)
        tag = hint.upper()
        matches = [f for f in all_files if tag in f.upper()]
        if matches:
            return matches[0]
        logger.warning(f"No GGUF file matching '{hint}' in repo '{repo_id}', auto-selecting")

    # Auto-select by preference order
    for pref in _QUANT_PREFERENCE:
        matches = [f for f in all_files if pref.upper() in f.upper()]
        if matches:
            logger.info(f"Auto-selected GGUF: {matches[0]}")
            return matches[0]

    # Fall back to first available
    logger.info(f"Using first available GGUF: {all_files[0]}")
    return all_files[0]


def _download_gguf(repo_id: str, filename: str) -> str:
    """Download a GGUF file from HuggingFace and return its local path."""
    from huggingface_hub import hf_hub_download

    logger.info(f"Downloading GGUF: {repo_id}/{filename}")
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    logger.info(f"GGUF downloaded: {path}")
    return path


# ---------------------------------------------------------------------------
# Top-level convenience function
# ---------------------------------------------------------------------------

def load_gguf_model(model_id: str) -> Tuple[GGUFLanguageModel, GGUFTokenizer]:
    """Load a GGUF model and return (model, tokenizer) ready for inference.

    This is the primary entry point for GGUF support in MLXGateway.
    """
    gguf_path, repo_id = resolve_gguf_file(model_id)
    logger.info(f"Loading GGUF weights from: {gguf_path}")

    weights, metadata = mx.load(gguf_path, return_metadata=True)
    arch = detect_architecture(metadata)
    logger.info(f"GGUF architecture: {arch}")

    if arch not in _SUPPORTED_ARCHITECTURES:
        logger.warning(
            f"Architecture '{arch}' is not explicitly tested. "
            f"Supported: {sorted(_SUPPORTED_ARCHITECTURES)}. Attempting to load anyway."
        )

    model = build_model_from_gguf(weights, metadata)
    tokenizer = GGUFTokenizer(metadata, repo_id=repo_id)
    return model, tokenizer
