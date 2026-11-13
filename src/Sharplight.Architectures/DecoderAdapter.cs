using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Generic decoder-only transformer (canonical arch <c>decoder</c>): Llama, Mistral, SmolLM, Qwen2/Qwen3, Gemma 2/3,
/// Phi-3 and friends are all this one graph with different hyper-parameters. Importers translate their format into
/// exactly the vocabulary below.
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>decoder.*</c>):
/// <list type="bullet">
/// <item><c>layer_count</c>, <c>hidden_size</c>, <c>ffn_hidden_size</c>, <c>context_length</c>, <c>norm_eps</c> (RMS norm epsilon).</item>
/// <item><c>attention.head_count</c>, <c>attention.kv_head_count</c> (GQA), <c>attention.head_dim</c>;
/// optional <c>attention.scale</c> (default 1/sqrt(head_dim)); <c>attention.logit_softcap</c> (tanh soft-capping of scores, 0 = off);
/// <c>attention.sliding_window</c> (tokens) with <c>attention.sliding_layers</c> (int[layer_count], 1 = layer uses the window).</item>
/// <item><c>rope.dim</c> (rotated dims, partial rotary when less than head_dim), <c>rope.base</c>, <c>rope.style</c>
/// ("neox" = rotate-half pairs (i, i+dim/2), the default; "gptj" = interleaved pairs), <c>rope.local_base</c> (base of sliding layers,
/// default rope.base), <c>rope.attention_scale</c> (cos/sin magnitude factor, e.g. yarn; folded into the score scale as its square).
/// Frequency scaling (llama3 / yarn / linear) is expressed as the optional weight <c>rope.freq_factors</c> (f32 [rope.dim/2], θ is divided by it)
/// which applies to the layers that are not sliding. Longrope (Phi-3) additionally carries <c>rope.freq_factors_long</c> and
/// <c>rope.original_context_length</c>: the long factors are used instead of the short ones when the configured context
/// (<see cref="AdapterOptions.ContextLength"/>, the model's <c>context_length</c> when unset) exceeds the original one. This is llama.cpp's
/// rule; HF switches on the current sequence length, which would leave already cached keys rotated with the other set, so a
/// KV-cached runtime picks once per model instance. Use a context &lt;= the original one for the short factors.
/// When <c>attention.sliding_window</c> is given without <c>attention.sliding_layers</c> (Mistral v0.1), every layer uses the window.</item>
/// <item><c>mlp.activation</c> ("silu" (default), "gelu", "gelu_tanh", "relu"), <c>mlp.gated</c> (default true: act(gate)·up).</item>
/// <item>Mixture of experts (a layer is MoE when it has <c>mlp.router.weight</c>): <c>moe.expert_count</c>, <c>moe.expert_used_count</c>
/// (top-k per token), <c>moe.normalize_weights</c> (renormalize the k selected router probabilities to sum 1; default false).
/// Router probabilities are softmax(router · h). Expert weights are rank-3 <c>mlp.experts.{gate,up,down}.weight</c> [E, N, K] and
/// <c>moe.ffn_hidden_size</c> is the experts' inner size (<c>ffn_hidden_size</c> then only describes dense layers).</item>
/// <item><c>embedding_scale</c> (token embeddings are multiplied by it, default 1; Gemma sqrt(hidden_size)),
/// <c>final_logit_softcap</c> (logits = cap·tanh(logits/cap), 0 = off).</item>
/// </list>
/// Weights: <c>token_embedding.weight</c> [V, H], <c>output_norm.weight</c>, <c>lm_head.weight</c> (optional, tied otherwise);
/// per layer <c>layers.{i}.</c>: <c>input_norm.weight</c> (in front of attention), <c>attention.{q,k,v,o}.weight</c> with optional
/// <c>.bias</c>, optional <c>attention.{q,k}_norm.weight</c> (per-head RMS norm before rope), optional <c>attention.out_norm.weight</c>
/// (norm of the attention output before the residual add), <c>post_attention_norm.weight</c> (in front of the MLP),
/// <c>mlp.{gate,up,down}.weight</c> (+ optional <c>.bias</c>; no gate when not gated), optional <c>mlp.out_norm.weight</c>.
/// Semantics: RMS norm weights multiply directly (any "1 + w" convention is folded in by the importer); q/k rows are in
/// rotate-half (neox) order per head, whatever <c>rope.style</c> says about the pairing the rope op uses on them is the
/// importer's concern (a llama.cpp "normal"-rope checkpoint is un-permuted on import so canonical weights are format independent).
/// </remarks>
public sealed class DecoderAdapter : IModelAdapter
{
    public string Name => "Decoder-only transformer";
    public IReadOnlyList<string> Architectures => ["decoder"];

    sealed record HParams(
        int NLayer, int NEmbd, int NFf, int NHead, int NHeadKv, int HeadDim, float Eps, float AttnScale, float SoftCap,
        int RopeDims, float RopeBase, float RopeLocalBase, string RopeStyle, int Window, int[] SlidingLayers,
        string Activation, bool Gated, float EmbeddingScale, float FinalSoftCap, int Vocab, int ContextLength,
        int Experts, int ExpertsUsed, bool NormalizeExpertWeights);

    // Longrope: the long factors replace the short ones when the configured context exceeds the original training context.
    static bool UseLongRope(int originalContext, int configuredContext) => originalContext > 0 && configuredContext > originalContext;

    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        int headDim = src.Arch<int>("attention.head_dim");
        float ropeAttn = src.Arch("rope.attention_scale", 1f);
        float ropeBase = src.Arch("rope.base", 10000f);
        var hp = new HParams(
            NLayer: src.Arch<int>("layer_count"),
            NEmbd: src.Arch<int>("hidden_size"),
            NFf: src.Arch<int>("ffn_hidden_size"),
            NHead: src.Arch<int>("attention.head_count"),
            NHeadKv: src.Arch("attention.kv_head_count", src.Arch<int>("attention.head_count")),
            HeadDim: headDim,
            Eps: src.Arch("norm_eps", 1e-5f),
            AttnScale: src.Arch("attention.scale", 1f / MathF.Sqrt(headDim)) * ropeAttn * ropeAttn,
            SoftCap: src.Arch("attention.logit_softcap", 0f),
            RopeDims: src.Arch("rope.dim", headDim),
            RopeBase: ropeBase,
            RopeLocalBase: src.Arch("rope.local_base", ropeBase),
            RopeStyle: src.Arch("rope.style", "neox"),
            Window: src.Arch("attention.sliding_window", 0),
            SlidingLayers: src.Arch("attention.sliding_layers", Array.Empty<int>()),
            Activation: src.Arch("mlp.activation", "silu"),
            Gated: src.Arch("mlp.gated", true),
            EmbeddingScale: src.Arch("embedding_scale", 1f),
            FinalSoftCap: src.Arch("final_logit_softcap", 0f),
            Vocab: src.Weight("token_embedding.weight")!.Shape[0].IsSymbolic ? 0 : (int)src.Weight("token_embedding.weight")!.Shape[0].Fixed,
            ContextLength: src.Arch("context_length", 32768),
            Experts: src.Arch("moe.expert_count", 0),
            ExpertsUsed: src.Arch("moe.expert_used_count", 0),
            NormalizeExpertWeights: src.Arch("moe.normalize_weights", false));
        if (hp.NHead % hp.NHeadKv != 0)
        {
            throw new InvalidDataException("decoder: attention.head_count must be a multiple of attention.kv_head_count.");
        }

        if (hp.SlidingLayers.Length != 0 && hp.SlidingLayers.Length != hp.NLayer)
        {
            throw new InvalidDataException("decoder: attention.sliding_layers must have layer_count entries.");
        }
        // Local rope (base / no frequency scaling) belongs to the interleaved sliding layers only; a model that declares a window
        // without a layer list (Mistral v0.1: sliding_window) applies it to every layer and keeps the ordinary rope.
        bool Sliding(int il) => hp.Window > 0 && hp.SlidingLayers.Length > 0 && hp.SlidingLayers[il] != 0;
        bool Windowed(int il) => hp.Window > 0 && (hp.SlidingLayers.Length == 0 || hp.SlidingLayers[il] != 0);

        int origCtx = src.Arch("rope.original_context_length", 0);
        int runtimeCtx = options.ContextLength > 0 ? options.ContextLength : hp.ContextLength;
        bool longRope = UseLongRope(origCtx, runtimeCtx);

        var states = new List<StateDefinition>();
        var kCache = new StateDefinition[hp.NLayer];
        var vCache = new StateDefinition[hp.NLayer];
        for (int il = 0; il < hp.NLayer; il++)
        {
            string g = $"layers.{il}";
            var shape = new Shape(Dim.Sym(WellKnown.ContextSymbol), hp.NHeadKv, hp.HeadDim);
            states.Add(kCache[il] = new StateDefinition { Name = $"{g}.k_cache", Type = options.KvCacheType, Shape = shape, Kind = StateKind.KeyCache, Group = g });
            states.Add(vCache[il] = new StateDefinition { Name = $"{g}.v_cache", Type = options.KvCacheType, Shape = shape, Kind = StateKind.ValueCache, Group = g });
        }

        var graphs = new Dictionary<string, ComputeGraph> { [WellKnown.DecodeGraph] = BuildTrunk(WellKnown.DecodeGraph, embeddingsInput: false) };
        if (options.IncludeAuxiliaryGraphs)
        {
            graphs["decode_embeddings"] = BuildTrunk("decode_embeddings", embeddingsInput: true);
        }
        // Packed-sequence graphs (embedding models such as Qwen3-Embedding, yes/no rerankers such as Qwen3-Reranker): causal attention
        // within each sequence, no KV cache. Last-token pooling unless the model declares otherwise.
        string pooling = PackedSequences.NormalizePooling(options.EmbeddingPooling) ?? PackedSequences.NormalizePooling(src.Get<string?>(CanonicalKeys.EmbeddingPooling, null)) ?? "last";
        bool normalize = options.EmbeddingNormalize ?? src.Get(CanonicalKeys.EmbeddingNormalize, true);
        if (options.IncludeAuxiliaryGraphs)
        {
            graphs[WellKnown.EmbedGraph] = BuildPacked(WellKnown.EmbedGraph, classify: false);
            graphs[WellKnown.ClassifyGraph] = BuildPacked(WellKnown.ClassifyGraph, classify: true);
        }

        var eos = new List<int>();
        if (src.Has(CanonicalKeys.TokenizerEosId))
        {
            eos.Add(src.Get<int>(CanonicalKeys.TokenizerEosId));
        }

        if (src.Has(CanonicalKeys.TokenizerEotId))
        {
            eos.Add(src.Get<int>(CanonicalKeys.TokenizerEotId));
        }

        return new ModelDefinition
        {
            Architecture = src.Architecture,
            Name = src.Get(CanonicalKeys.Name, src.Architecture),
            Weights = src.Weights,
            Graphs = graphs,
            States = states,
            SourceMetadata = src.Metadata,
            OwnedResources = src.OwnedResources,
            Properties = new Dictionary<string, object>
            {
                [WellKnown.VocabSize] = hp.Vocab,
                [WellKnown.ContextLength] = hp.ContextLength,
                [WellKnown.EmbeddingLength] = hp.NEmbd,
                [WellKnown.PositionAxes] = 1,
                [WellKnown.EosTokenIds] = eos.ToArray(),
                ["n_layer"] = hp.NLayer,
                [WellKnown.EmbeddingPooling] = pooling,
                [WellKnown.EmbeddingNormalize] = normalize,
                [WellKnown.CausalAttention] = true,
                [WellKnown.ClassifierLabels] = hp.Vocab,
            },
        };

        ComputeGraph BuildPacked(string name, bool classify)
        {
            var b = new GraphBuilder(name, src.Weight);
            var inp = PackedSequences.Declare(b, tokenTypes: false);
            Value x;
            using (b.Group("input"))
            {
                x = b.Gather(b.Weight("token_embedding.weight"), inp.Tokens, "inp_embd");
                if (hp.EmbeddingScale != 1f)
                {
                    x = b.Scale(x, hp.EmbeddingScale);
                }
            }
            var freqFactors = b.TryWeight("rope.freq_factors");
            if (longRope && b.TryWeight("rope.freq_factors_long") is { } longFactors)
            {
                freqFactors = longFactors;
            }

            x = Layers(b, x, inp.Positions, freqFactors, inp.RowSpans);
            using (b.Group("output"))
            {
                if (!classify)
                {
                    var hidden = b.RmsNorm(x, b.Weight("output_norm.weight"), hp.Eps, "result_norm");
                    PackedSequences.EmbeddingsOutput(b, hidden, inp.SequenceSpans, pooling, normalize);
                }
                else
                {
                    // Score heads read the language model head at each sequence's last row (e.g. "yes" / "no" logits).
                    var last = PackedSequences.Pool(b, x, inp.SequenceSpans, "last", "last_rows");
                    var hidden = b.RmsNorm(last, b.Weight("output_norm.weight"), hp.Eps, "result_norm");
                    var head = b.TryWeight("lm_head.weight") ?? b.Weight("token_embedding.weight");
                    var logits = b.MatMul(hidden, head, "logits");
                    if (hp.FinalSoftCap > 0)
                    {
                        logits = b.Scale(b.Unary("Tanh", b.Scale(logits, 1f / hp.FinalSoftCap)), hp.FinalSoftCap, 0, "logits_capped");
                    }

                    b.Output(WellKnown.LogitsOutput, logits);
                }
            }
            return b.Build();
        }

        Value Layers(GraphBuilder b, Value x, Value positions, Value? freqFactors, Value? rowSpans)
        {
            for (int il = 0; il < hp.NLayer; il++)
            {
                using var _ = b.Group($"layers.{il}");
                string p = $"layers.{il}.";
                var residual = x;
                var h = b.RmsNorm(x, b.Weight(p + "input_norm.weight"), hp.Eps, $"{p}attn_norm");
                var attnOut = Attention(b, h, p, il, positions, freqFactors, rowSpans);
                if (b.TryWeight(p + "attention.out_norm.weight") is { } on1)
                {
                    attnOut = b.RmsNorm(attnOut, on1, hp.Eps);
                }

                x = b.Add(attnOut, residual, $"{p}attn_residual");

                var ffnResidual = x;
                h = b.RmsNorm(x, b.Weight(p + "post_attention_norm.weight"), hp.Eps, $"{p}post_attn_norm");
                var ffn = b.HasWeight(p + "mlp.router.weight") ? MoeMlp(b, h, p) : Mlp(b, h, p);
                if (b.TryWeight(p + "mlp.out_norm.weight") is { } on2)
                {
                    ffn = b.RmsNorm(ffn, on2, hp.Eps);
                }

                x = b.Add(ffn, ffnResidual, $"{p}l_out");
            }
            return x;
        }

        ComputeGraph BuildTrunk(string name, bool embeddingsInput)
        {
            var b = new GraphBuilder(name, src.Weight);
            var T = Dim.Sym(WellKnown.TokenSymbol);
            Value x;
            using (b.Group("input"))
            {
                if (embeddingsInput)
                {
                    x = b.Input("embeddings", InputRole.Embeddings, DataType.F32, new Shape(T, hp.NEmbd));
                }
                else
                {
                    var tokens = b.Input("tokens", InputRole.TokenIds, DataType.I32, new Shape(T));
                    x = b.Gather(b.Weight("token_embedding.weight"), tokens, "inp_embd");
                    if (hp.EmbeddingScale != 1f)
                    {
                        x = b.Scale(x, hp.EmbeddingScale);
                    }
                }
            }
            var positions = b.Input("positions", InputRole.Positions, DataType.I32, new Shape(T, 1));
            var outRows = b.Input("output_rows", InputRole.OutputRows, DataType.I32, new Shape(Dim.Sym("R")));
            var freqFactors = b.TryWeight("rope.freq_factors");
            if (longRope && b.TryWeight("rope.freq_factors_long") is { } longFactors)
            {
                freqFactors = longFactors;
            }

            x = Layers(b, x, positions, freqFactors, rowSpans: null);

            using (b.Group("output"))
            {
                var hidden = b.RmsNorm(x, b.Weight("output_norm.weight"), hp.Eps, "result_norm");
                var rows = b.Gather(hidden, outRows, "selected_rows");
                var head = b.TryWeight("lm_head.weight") ?? b.Weight("token_embedding.weight");
                var logits = b.MatMul(rows, head, "logits");
                if (hp.FinalSoftCap > 0)
                {
                    logits = b.Scale(b.Unary("Tanh", b.Scale(logits, 1f / hp.FinalSoftCap)), hp.FinalSoftCap, 0, "logits_capped");
                }

                b.Output("logits", logits);
                b.Output("hidden", hidden);
            }
            return b.Build();
        }

        Value Attention(GraphBuilder b, Value h, string p, int il, Value positions, Value? freqFactors, Value? rowSpans)
        {
            int D = hp.HeadDim;
            var q = b.Linear(h, p + "attention.q.weight", p + "attention.q.bias");
            var k = b.Linear(h, p + "attention.k.weight", p + "attention.k.bias");
            var v = b.Reshape(b.Linear(h, p + "attention.v.weight", p + "attention.v.bias"), 0, hp.NHeadKv, D);
            // q/k norms: over the whole projection (weight has heads*D entries) or per head (D entries), before the head split.
            if (b.TryWeight(p + "attention.q_norm.weight") is { } qn && qn.Shape[0].Fixed != D)
            {
                q = b.RmsNorm(q, qn, hp.Eps);
            }

            if (b.TryWeight(p + "attention.k_norm.weight") is { } kn && kn.Shape[0].Fixed != D)
            {
                k = b.RmsNorm(k, kn, hp.Eps);
            }

            q = b.Reshape(q, 0, hp.NHead, D);
            k = b.Reshape(k, 0, hp.NHeadKv, D);
            if (b.TryWeight(p + "attention.q_norm.weight") is { } qn2 && qn2.Shape[0].Fixed == D)
            {
                q = b.RmsNorm(q, qn2, hp.Eps);
            }

            if (b.TryWeight(p + "attention.k_norm.weight") is { } kn2 && kn2.Shape[0].Fixed == D)
            {
                k = b.RmsNorm(k, kn2, hp.Eps);
            }

            bool sliding = Sliding(il);
            var rope = Attributes.From(("dims", hp.RopeDims), ("base", sliding ? hp.RopeLocalBase : hp.RopeBase), ("style", hp.RopeStyle));
            Value[] qIn = freqFactors is { } ff && !sliding ? [q, positions, ff] : [q, positions];
            Value[] kIn = freqFactors is { } ff2 && !sliding ? [k, positions, ff2] : [k, positions];
            q = b.Emit1("Rope", qIn, rope);
            k = b.Emit1("Rope", kIn, rope);
            if (rowSpans is not null)
            {
                // Packed sequences: causal within each sequence; sliding windows are not applied (inputs are short).
                var packed = PackedSequences.Attention(b, q, k, v, rowSpans, hp.AttnScale, causal: true, hp.SoftCap, $"{p}attn");
                return b.Linear(b.Reshape(packed, 0, hp.NHead * D), p + "attention.o.weight", p + "attention.o.bias", $"{p}attn_out");
            }

            b.CacheWrite(b.State(kCache[il]), k);
            b.CacheWrite(b.State(vCache[il]), v);
            var attrs = new List<(string, object)> { ("scale", hp.AttnScale), ("offset", WellKnown.PastScalar) };
            if (Windowed(il))
            {
                attrs.Add(("window", hp.Window));
            }

            if (hp.SoftCap > 0)
            {
                attrs.Add(("softcap", hp.SoftCap));
            }

            b.RuntimeScalar(WellKnown.PastScalar);
            var attn = b.Emit1("CachedAttention", [q, b.State(kCache[il]), b.State(vCache[il])], Attributes.From([.. attrs]), $"{p}attn");
            return b.Linear(b.Reshape(attn, 0, hp.NHead * D), p + "attention.o.weight", p + "attention.o.bias", $"{p}attn_out");
        }

        Value ActOf(GraphBuilder b, Value t) => hp.Activation switch
        {
            "silu" => b.Silu(t),
            "gelu" => b.Unary("Gelu", t),
            "gelu_tanh" => b.Unary("GeluTanh", t),
            "relu" => b.Unary("Relu", t),
            _ => throw new NotSupportedException($"decoder: unknown mlp.activation '{hp.Activation}'."),
        };

        // Softmax router, top-k experts per token, expert-indexed gate/up/down products, weighted sum over the k slots.
        Value MoeMlp(GraphBuilder b, Value h, string p)
        {
            if (hp.Experts <= 0 || hp.ExpertsUsed <= 0)
            {
                throw new InvalidDataException("decoder: MoE weights present but moe.expert_count / moe.expert_used_count are missing.");
            }

            var probs = b.Emit1("Softmax", [b.Linear(h, p + "mlp.router.weight")], null, $"{p}router_probs");
            var top = b.Emit("TopK", [probs], Attributes.From(("k", (long)hp.ExpertsUsed)), $"{p}router_topk");
            Value weights = top[0], ids = top[1];
            if (hp.NormalizeExpertWeights)
            {
                weights = b.Binary("Div", weights, b.Emit1("ReduceSum", [weights]));
            }

            var up = b.Emit1("MatMulId", [h, b.Weight(p + "mlp.experts.up.weight"), ids], null, $"{p}experts_up");
            var gate = b.Emit1("MatMulId", [h, b.Weight(p + "mlp.experts.gate.weight"), ids], null, $"{p}experts_gate");
            var act = b.Mul(ActOf(b, gate), up);
            var down = b.Emit1("MatMulId", [act, b.Weight(p + "mlp.experts.down.weight"), ids], null, $"{p}experts_down");   // [T, k, H]
            var weighted = b.Mul(down, b.Reshape(weights, 0, hp.ExpertsUsed, 1));
            var summed = b.Emit1("ReduceSum", [b.Permute(weighted, 0, 2, 1)]);                                              // [T, H, 1]
            return b.Reshape(summed, 0, hp.NEmbd);
        }

        Value Mlp(GraphBuilder b, Value h, string p)
        {
            Value Act(Value t) => ActOf(b, t);
            var up = b.Linear(h, p + "mlp.up.weight", p + "mlp.up.bias");
            var act = hp.Gated ? b.Mul(Act(b.Linear(h, p + "mlp.gate.weight", p + "mlp.gate.bias")), up) : Act(up);
            return b.Linear(act, p + "mlp.down.weight", p + "mlp.down.bias", $"{p}ffn_out");
        }
    }
}
