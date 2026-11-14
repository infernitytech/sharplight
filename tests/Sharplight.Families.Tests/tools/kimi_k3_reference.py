"""
Kimi K3 / K3.1 reference model on CPU in float32: the published remote code (modeling_kimi_linear.py and configuration_kimi_k3.py of a Kimi-K3
checkpoint) with the parts of the flash-linear-attention package it imports written out in plain torch (short causal convolutions, the gated
RMS norm and the Kimi delta rule as a token loop). K3.1 ("gated DSA", llama.cpp kimi-k31) has no published modeling code: with a dsa_config the
latent-attention layers get the lightning indexer of llama.cpp's graph (q from the normalized compressed query, LayerNorm'd shared key, per-head
weights scaled by 1/sqrt(heads * dims), no rotary embedding; score = sum_h w_h * relu(q_h . k)) and attend only to the index_topk best visible
positions (all of them while a row sees at most index_topk). As in llama.cpp, every latent-attention layer is sparse: one that is not in dsa_layers
(GGUF attention.indexer.types false) has no indexer and reuses the selection of the last layer that has one. Used by mk_tiny_kimi_k3.py.

    from kimi_k3_reference import load_text_model
    model = load_text_model(<remote code dir>, text_config dict, state dict or None)
"""
import importlib.util, math, sys, types
import torch
import torch.nn.functional as F
from torch import nn


def _install_fla_stub():
    """Registers a minimal 'fla' package with the names modeling_kimi_linear.py imports."""
    if "fla" in sys.modules:
        return

    class ShortConvolution(nn.Conv1d):
        def __init__(self, hidden_size, kernel_size, activation=None, bias=False, **_):
            super().__init__(hidden_size, hidden_size, kernel_size, groups=hidden_size, bias=bias, padding=0)
            self.activation = activation

        def forward(self, x, cache=None, output_final_state=False, cu_seqlens=None, **_):
            assert cu_seqlens is None and cache is None
            t = x.shape[1]
            y = F.conv1d(F.pad(x.transpose(1, 2), (self.kernel_size[0] - 1, 0)), self.weight, self.bias, groups=self.groups)[..., :t]
            y = y.transpose(1, 2)
            if self.activation == "silu":
                y = F.silu(y)
            return y, None

    class FusedRMSNormGated(nn.Module):
        def __init__(self, hidden_size, eps=1e-5, activation="sigmoid", **_):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.eps = eps
            assert activation == "sigmoid"

        def forward(self, x, g):
            x = x.float()
            normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.float()
            return normed * torch.sigmoid(g.float())

    def l2norm(x, eps=1e-6):
        return x / torch.sqrt((x * x).sum(-1, keepdim=True) + eps)

    def kda(q, k, v, g, beta, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
            use_beta_sigmoid_in_kernel=False, cu_seqlens=None, **_):
        # g is the final log decay [B, T, H, D] (K3 computes it outside the kernel); state S [B, H, Dk, Dv].
        assert not use_gate_in_kernel and cu_seqlens is None and initial_state is None
        q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
        if use_qk_l2norm_in_kernel:
            q, k = l2norm(q), l2norm(k)
        if use_beta_sigmoid_in_kernel:
            beta = torch.sigmoid(beta)
        b, t, h, dk = k.shape
        q = q * dk ** -0.5
        state = torch.zeros(b, h, dk, v.shape[-1])
        out = torch.zeros(b, t, h, v.shape[-1])
        for i in range(t):
            state = state * g[:, i].exp()[..., None]
            delta = (v[:, i] - (state * k[:, i][..., None]).sum(-2)) * beta[:, i][..., None]
            state = state + k[:, i][..., None] * delta[..., None, :]
            out[:, i] = (state * q[:, i][..., None]).sum(-2)
        return out, state

    fla = types.ModuleType("fla")
    modules = types.ModuleType("fla.modules")
    modules.FusedRMSNormGated, modules.ShortConvolution = FusedRMSNormGated, ShortConvolution
    ops = types.ModuleType("fla.ops")
    ops_kda = types.ModuleType("fla.ops.kda")
    ops_kda.chunk_kda = ops_kda.fused_recurrent_kda = kda
    ops_utils = types.ModuleType("fla.ops.utils")
    ops_index = types.ModuleType("fla.ops.utils.index")
    ops_index.prepare_cu_seqlens_from_mask = ops_index.prepare_lens_from_mask = None
    utils = types.ModuleType("fla.utils")
    utils.tensor_cache = lambda f: f
    for name, module in {"fla": fla, "fla.modules": modules, "fla.ops": ops, "fla.ops.kda": ops_kda, "fla.ops.utils": ops_utils,
                         "fla.ops.utils.index": ops_index, "fla.utils": utils}.items():
        sys.modules[name] = module


def _load_remote(code_dir):
    """Imports the remote modeling code of a Kimi-K3 checkpoint as package 'kimi_k3_remote' (without writing bytecode next to it)."""
    _install_fla_stub()
    sys.dont_write_bytecode = True
    if "kimi_k3_remote.modeling_kimi_linear" in sys.modules:
        return sys.modules["kimi_k3_remote.modeling_kimi_linear"]
    package = types.ModuleType("kimi_k3_remote")
    package.__path__ = [code_dir]
    sys.modules["kimi_k3_remote"] = package
    for name in ("configuration_kimi_k3", "modeling_kimi_linear"):
        spec = importlib.util.spec_from_file_location(f"kimi_k3_remote.{name}", f"{code_dir}/{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return sys.modules["kimi_k3_remote.modeling_kimi_linear"]


class LightningIndexer(nn.Module):
    """K3.1's DSA indexer as llama.cpp's kimi-k31 graph computes it (no rotary embedding)."""

    def __init__(self, config, dsa):
        super().__init__()
        self.heads, self.dims, self.topk = dsa["index_num_heads"], dsa["index_head_dim"], dsa["index_topk"]
        self.wq_b = nn.Linear(config.q_lora_rank, self.heads * self.dims, bias=False)
        self.wk = nn.Linear(config.hidden_size, self.dims, bias=False)
        self.k_norm = nn.LayerNorm(self.dims, eps=dsa.get("index_norm_eps", config.rms_norm_eps))
        self.weights_proj = nn.Linear(config.hidden_size, self.heads, bias=False)

    def scores(self, x, q_latent):
        """Index scores [B, T, T] (-inf above the diagonal)."""
        b, t, _ = x.shape
        q = self.wq_b(q_latent).view(b, t, self.heads, self.dims)
        k = self.k_norm(self.wk(x))                                                  # [B, T, D]
        w = self.weights_proj(x) / math.sqrt(self.heads * self.dims)                 # [B, T, H]
        s = (torch.relu(torch.einsum("bthd,bsd->bths", q, k)) * w[..., None]).sum(2)
        return s.masked_fill(torch.ones(t, t, dtype=torch.bool).triu(1), float("-inf"))

    def selection_mask(self, scores):
        """Additive mask [B, 1, T, T]: 0 at the topk best visible positions of every row, -inf elsewhere."""
        t = scores.shape[-1]
        mask = torch.full_like(scores, float("-inf"))
        for row in range(t):
            visible = row + 1
            keep = torch.arange(visible) if visible <= self.topk else scores[:, row, :visible].topk(self.topk, dim=-1).indices[0]
            mask[:, row, keep] = 0
        return mask[:, None]


def _dsa_forward(original):
    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_values=None, **kwargs):
        if getattr(self, "indexer", None) is None:
            if getattr(self, "shares", None) is None:
                return original(self, hidden_states, attention_mask, position_ids, past_key_values, **kwargs)
            # a latent-attention layer without an indexer reuses the selection of the last layer that has one
            return original(self, hidden_states, self.shares.selection, position_ids, past_key_values, **kwargs)
        assert past_key_values is None
        q_latent = self.q_a_layernorm(self.q_a_proj(hidden_states))
        scores = self.indexer.scores(hidden_states, q_latent)
        if self.record is not None:
            self.record.append(scores.detach())
        self.selection = self.indexer.selection_mask(scores)
        return original(self, hidden_states, self.selection, position_ids, past_key_values, **kwargs)
    return forward


def load_text_model(code_dir, text_config, state=None):
    """KimiLinearForCausalLM (float32, eager attention) for a K3 text config; K3.1 when it has a dsa_config."""
    remote = _load_remote(code_dir)
    config = remote.KimiLinearConfig(**text_config)
    config._attn_implementation = "eager"
    dsa = text_config.get("dsa_config")
    if dsa is not None and not getattr(remote.KimiMLAAttention, "_dsa_patched", False):
        remote.KimiMLAAttention.forward = _dsa_forward(remote.KimiMLAAttention.forward)
        remote.KimiMLAAttention._dsa_patched = True
    model = remote.KimiLinearForCausalLM(config)
    last_indexed = None
    for i, layer in enumerate(model.model.layers):
        attention = layer.self_attn
        attention.record = None
        attention.indexer = LightningIndexer(config, dsa) if dsa is not None and (i + 1) in dsa["dsa_layers"] else None
        if attention.indexer is not None:
            last_indexed = attention
        elif dsa is not None and (i + 1) in config.linear_attn_config["full_attn_layers"]:
            assert last_indexed is not None, f"latent-attention layer {i + 1} has no indexer and no earlier layer has one"
            object.__setattr__(attention, "shares", last_indexed)   # a plain reference, not a submodule
    if state is not None:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise ValueError(f"missing {missing[:5]}, unexpected {unexpected[:5]}")
    return model.float().eval()
