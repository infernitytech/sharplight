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
/// which applies to the layers that are not sliding.</item>
/// <item><c>mlp.activation</c> ("silu" (default), "gelu", "gelu_tanh", "relu"), <c>mlp.gated</c> (default true: act(gate)·up).</item>
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
        string Activation, bool Gated, float EmbeddingScale, float FinalSoftCap, int Vocab, int ContextLength);

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
            ContextLength: src.Arch("context_length", 32768));
        if (hp.NHead % hp.NHeadKv != 0) throw new InvalidDataException("decoder: attention.head_count must be a multiple of attention.kv_head_count.");
        if (hp.SlidingLayers.Length != 0 && hp.SlidingLayers.Length != hp.NLayer) throw new InvalidDataException("decoder: attention.sliding_layers must have layer_count entries.");
        bool Sliding(int il) => hp.Window > 0 && hp.SlidingLayers.Length > 0 && hp.SlidingLayers[il] != 0;

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
        if (options.IncludeAuxiliaryGraphs) graphs["decode_embeddings"] = BuildTrunk("decode_embeddings", embeddingsInput: true);

        var eos = new List<int>();
        if (src.Has(CanonicalKeys.TokenizerEosId)) eos.Add(src.Get<int>(CanonicalKeys.TokenizerEosId));
        if (src.Has(CanonicalKeys.TokenizerEotId)) eos.Add(src.Get<int>(CanonicalKeys.TokenizerEotId));

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
            },
        };

        ComputeGraph BuildTrunk(string name, bool embeddingsInput)
        {
            var b = new GraphBuilder(name, src.Weight);
            var T = Dim.Sym(WellKnown.TokenSymbol);
            Value x;
            using (b.Group("input"))
            {
                if (embeddingsInput)
                    x = b.Input("embeddings", InputRole.Embeddings, DataType.F32, new Shape(T, hp.NEmbd));
                else
                {
                    var tokens = b.Input("tokens", InputRole.TokenIds, DataType.I32, new Shape(T));
                    x = b.Gather(b.Weight("token_embedding.weight"), tokens, "inp_embd");
                    if (hp.EmbeddingScale != 1f) x = b.Scale(x, hp.EmbeddingScale);
                }
            }
            var positions = b.Input("positions", InputRole.Positions, DataType.I32, new Shape(T, 1));
            var outRows = b.Input("output_rows", InputRole.OutputRows, DataType.I32, new Shape(Dim.Sym("R")));
            var freqFactors = b.TryWeight("rope.freq_factors");

            for (int il = 0; il < hp.NLayer; il++)
            {
                using var _ = b.Group($"layers.{il}");
                string p = $"layers.{il}.";
                var residual = x;
                var h = b.RmsNorm(x, b.Weight(p + "input_norm.weight"), hp.Eps, $"{p}attn_norm");
                var attnOut = Attention(b, h, p, il, positions, freqFactors);
                if (b.TryWeight(p + "attention.out_norm.weight") is { } on1) attnOut = b.RmsNorm(attnOut, on1, hp.Eps);
                x = b.Add(attnOut, residual, $"{p}attn_residual");

                var ffnResidual = x;
                h = b.RmsNorm(x, b.Weight(p + "post_attention_norm.weight"), hp.Eps, $"{p}post_attn_norm");
                var ffn = Mlp(b, h, p);
                if (b.TryWeight(p + "mlp.out_norm.weight") is { } on2) ffn = b.RmsNorm(ffn, on2, hp.Eps);
                x = b.Add(ffn, ffnResidual, $"{p}l_out");
            }

            using (b.Group("output"))
            {
                var hidden = b.RmsNorm(x, b.Weight("output_norm.weight"), hp.Eps, "result_norm");
                var rows = b.Gather(hidden, outRows, "selected_rows");
                var head = b.TryWeight("lm_head.weight") ?? b.Weight("token_embedding.weight");
                var logits = b.MatMul(rows, head, "logits");
                if (hp.FinalSoftCap > 0)
                    logits = b.Scale(b.Unary("Tanh", b.Scale(logits, 1f / hp.FinalSoftCap)), hp.FinalSoftCap, 0, "logits_capped");
                b.Output("logits", logits);
                b.Output("hidden", hidden);
            }
            return b.Build();
        }

        Value Attention(GraphBuilder b, Value h, string p, int il, Value positions, Value? freqFactors)
        {
            int D = hp.HeadDim;
            var q = b.Reshape(b.Linear(h, p + "attention.q.weight", p + "attention.q.bias"), 0, hp.NHead, D);
            var k = b.Reshape(b.Linear(h, p + "attention.k.weight", p + "attention.k.bias"), 0, hp.NHeadKv, D);
            var v = b.Reshape(b.Linear(h, p + "attention.v.weight", p + "attention.v.bias"), 0, hp.NHeadKv, D);
            if (b.TryWeight(p + "attention.q_norm.weight") is { } qn) q = b.RmsNorm(q, qn, hp.Eps);
            if (b.TryWeight(p + "attention.k_norm.weight") is { } kn) k = b.RmsNorm(k, kn, hp.Eps);

            bool sliding = Sliding(il);
            var rope = Attributes.From(("dims", hp.RopeDims), ("base", sliding ? hp.RopeLocalBase : hp.RopeBase), ("style", hp.RopeStyle));
            Value[] qIn = freqFactors is { } ff && !sliding ? [q, positions, ff] : [q, positions];
            Value[] kIn = freqFactors is { } ff2 && !sliding ? [k, positions, ff2] : [k, positions];
            q = b.Emit1("Rope", qIn, rope);
            k = b.Emit1("Rope", kIn, rope);

            b.CacheWrite(b.State(kCache[il]), k);
            b.CacheWrite(b.State(vCache[il]), v);
            var attrs = new List<(string, object)> { ("scale", hp.AttnScale), ("offset", WellKnown.PastScalar) };
            if (sliding) attrs.Add(("window", hp.Window));
            if (hp.SoftCap > 0) attrs.Add(("softcap", hp.SoftCap));
            b.RuntimeScalar(WellKnown.PastScalar);
            var attn = b.Emit1("CachedAttention", [q, b.State(kCache[il]), b.State(vCache[il])], Attributes.From([.. attrs]), $"{p}attn");
            return b.Linear(b.Reshape(attn, 0, hp.NHead * D), p + "attention.o.weight", p + "attention.o.bias", $"{p}attn_out");
        }

        Value Mlp(GraphBuilder b, Value h, string p)
        {
            Value Act(Value t) => hp.Activation switch
            {
                "silu" => b.Silu(t),
                "gelu" => b.Unary("Gelu", t),
                "gelu_tanh" => b.Unary("GeluTanh", t),
                "relu" => b.Unary("Relu", t),
                _ => throw new NotSupportedException($"decoder: unknown mlp.activation '{hp.Activation}'."),
            };
            var up = b.Linear(h, p + "mlp.up.weight", p + "mlp.up.bias");
            var act = hp.Gated ? b.Mul(Act(b.Linear(h, p + "mlp.gate.weight", p + "mlp.gate.bias")), up) : Act(up);
            return b.Linear(act, p + "mlp.down.weight", p + "mlp.down.bias", $"{p}ffn_out");
        }
    }
}
