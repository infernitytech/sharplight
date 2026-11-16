using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Multi-pass hybrid decoder with parallel residual streams and a per-token depth memory (canonical arch <c>depth_memory</c>; Metis 1.6): a stack of
/// Mamba-2 and attention layers that runs P times over S residual streams, every pass with its own stream mixing, its own caches and states, and a
/// per-pass low-rank adapter on the attention projections. Each layer's feed-forward block is a mixture of experts in a latent space whose router
/// also reads routing features (from the stream state, the memory and the routing history); the memory is written after every attention layer and
/// at the end of every pass, together with a continuation estimate.
/// </summary>
/// <remarks>
/// Per virtual layer v = p·L + i (pass p, physical layer i, weights <c>layers.{i}.</c>, states <c>layers.{v}</c>):
/// <list type="number">
/// <item>With a non-empty <see cref="DepthMemory"/>: every stream reads it (gated) and r = the stream mean of the retrieved projections.</item>
/// <item>Routing features f = <see cref="DepthMemory.RouteFeatures"/>(mean(x), r, –, history).</item>
/// <item>Mixer through <see cref="StaticStreamMixing"/> (<c>streams.attention.*</c>, norm <c>input_norm.weight</c>): Mamba-2
/// (<see cref="Mamba2Layer"/>, <c>ssm.*</c>) or attention without positions from the fused <c>attention.qkv.weight</c> [(Hq + 2·Hkv)·D, H] plus
/// <c>attention.pass_gate</c>[p] · <c>attention.pass_lora.up.weight</c> · <c>attention.pass_lora.down.weight</c> · input, output <c>attention.o.weight</c>.</item>
/// <item>Feed-forward through <see cref="StaticStreamMixing"/> (<c>streams.mlp.*</c>, norm <c>post_attention_norm.weight</c>):
/// latent = <c>latent.down.weight</c> · input; the experts (<see cref="MoeBlock"/> below <c>mlp.</c>, on the latent rows, with a shared expert)
/// are routed by <c>mlp.router.weight</c> [E, L + R] on [latent; f] with the selection bias <c>mlp.router_bias</c>; output
/// <c>latent.up.weight</c> · experts. The routing summary s = <see cref="MoeRoutingSignals.Summary"/> with <c>mlp.expert_embedding</c> [E, R];
/// history = ½·history + ½·s.</item>
/// <item>After an attention layer: a memory entry from mean(x) and [s, entropy, confidence, c] (c: the previous pass's continuation, 0 in the first).</item>
/// </list>
/// At the end of every pass: a memory read, the <see cref="DepthContinuationHead"/> on the stream mean, its change over the pass and the routing
/// features, and (except after the last pass) a memory entry with the new continuation; entropy and confidence come from the pass's last layer.
/// The streams start as the token embedding plus <c>streams.embedding</c> [S, H]; the output is the stream mean through <c>output_norm.weight</c>
/// and <c>lm_head.weight</c> (tied to the embedding when absent).
/// Hyper-parameters (<c>depth_memory.*</c>): <c>layer_count</c> (physical layers), <c>pass_count</c>, <c>hidden_size</c>, <c>norm_eps</c>,
/// <c>context_length</c>, <c>ssm.*</c> with <c>ssm.norm_eps</c>, <c>attention.*</c>, <c>moe.*</c>, <c>continuation.logit_limit</c>.
/// </remarks>
public sealed class DepthMemoryAdapter : IModelAdapter
{
    /// <inheritdoc/>
    public string Name => "Depth memory (multi-pass Mamba-2 / attention streams with latent experts)";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => ["depth_memory"];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        int layerCount = src.Arch<int>("layer_count");
        int passes = src.Arch("pass_count", 1);
        int hidden = src.Arch<int>("hidden_size");
        float eps = src.Arch("norm_eps", 1e-5f);
        float limit = src.Arch("continuation.logit_limit", float.MaxValue);
        int streams = (int)Required(src, "streams.embedding").Shape[0].Fixed;
        int memoryWidth = (int)Required(src, "depth_memory.query.weight").Shape[0].Fixed;
        int latent = (int)Required(src, "layers.0.latent.down.weight").Shape[0].Fixed;
        var mamba = new Mamba2Layer(src, src.Arch("ssm.norm_eps", eps));
        var attention = new HybridAttention(src);
        var moe = MoeOptions.Read(src);
        var activation = MlpActivations.Create("silu");

        var isAttention = new bool[layerCount];
        var states = new List<StateDefinition>();
        var layerStates = new (StateDefinition First, StateDefinition Second)[layerCount * passes];
        for (int il = 0; il < layerCount; il++)
        {
            isAttention[il] = src.Weight($"layers.{il}.attention.qkv.weight") is not null;
        }

        for (int v = 0; v < layerCount * passes; v++)
        {
            layerStates[v] = isAttention[v % layerCount]
                ? attention.DefineStates($"layers.{v}", options.KvCacheType)
                : mamba.DefineStates($"layers.{v}");
            states.Add(layerStates[v].First);
            states.Add(layerStates[v].Second);
        }

        var graphs = new Dictionary<string, ComputeGraph> { [WellKnown.DecodeGraph] = BuildTrunk(WellKnown.DecodeGraph, false) };
        if (options.IncludeAuxiliaryGraphs)
        {
            graphs["decode_embeddings"] = BuildTrunk("decode_embeddings", true);
        }

        return RecurrentModelDefinition.Create(src, layerCount * passes, hidden, graphs, states);

        ComputeGraph BuildTrunk(string name, bool embeddingsInput)
        {
            var b = new GraphBuilder(name, src.Weight);
            var T = Dim.Sym(WellKnown.TokenSymbol);
            Value embeddings;
            using (b.Group("input"))
            {
                embeddings = embeddingsInput
                    ? b.Input("embeddings", InputRole.Embeddings, DataType.F32, new Shape(T, hidden))
                    : b.Gather(b.Weight("token_embedding.weight"), b.Input("tokens", InputRole.TokenIds, DataType.I32, new Shape(T)), "inp_embd");
            }

            var positions = b.Input("positions", InputRole.Positions, DataType.I32, new Shape(T, 1));
            var outRows = b.Input("output_rows", InputRole.OutputRows, DataType.I32, new Shape(Dim.Sym("R")));
            var x = b.Add(b.Reshape(embeddings, 0, 1, hidden), b.Weight("streams.embedding"), "streams_in");                       // [T, S, H]
            var memory = new DepthMemory(streams, hidden, memoryWidth);
            var previous = StaticStreamMixing.Mean(b, x, streams, hidden);
            Value? history = null;
            Value? continuation = null;
            for (int pass = 0; pass < passes; pass++)
            {
                Value summary = null!, entropy = null!, confidence = null!;
                for (int il = 0; il < layerCount; il++)
                {
                    int v = pass * layerCount + il;
                    using var _ = b.Group($"layers.{v}");
                    string p = $"layers.{il}.";
                    bool writes = isAttention[il] || il == layerCount - 1;
                    Value? retrieved = null;
                    if (!memory.IsEmpty)
                    {
                        (x, retrieved) = memory.Read(b, x);
                    }

                    var features = DepthMemory.RouteFeatures(b, StaticStreamMixing.Mean(b, x, streams, hidden), retrieved, null, history);
                    var (mixed, input) = StaticStreamMixing.Read(b, x, p + "streams.attention.", pass, streams, hidden, b.Weight(p + "input_norm.weight"), eps);
                    var mixer = isAttention[il]
                        ? Attention(b, input, positions, pass, v, p)
                        : mamba.Build(b, input, p, layerStates[v].First, layerStates[v].Second);
                    x = StaticStreamMixing.Write(b, mixed, p + "streams.attention.", pass, streams, hidden, mixer);

                    (mixed, input) = StaticStreamMixing.Read(b, x, p + "streams.mlp.", pass, streams, hidden, b.Weight(p + "post_attention_norm.weight"), eps);
                    var rows = b.Linear(input, p + "latent.down.weight", name: $"{p}latent_in");
                    var (weights, ids, logits) = MoeRouter.RouteWithLogits(b, b.Concat(1, rows, features), p + "mlp.", moe);
                    var experts = MoeBlock.Build(b, rows, p, "mlp.", moe, latent, activation, route: (weights, ids));
                    var ffn = b.Linear(experts, p + "latent.up.weight", name: $"{p}ffn_out");
                    summary = MoeRoutingSignals.Summary(b, b.Weight(p + "mlp.expert_embedding"), weights, ids, moe.ExpertsUsed);
                    if (writes)
                    {
                        (entropy, confidence) = MoeRoutingSignals.EntropyAndConfidence(b, logits);
                    }

                    x = StaticStreamMixing.Write(b, mixed, p + "streams.mlp.", pass, streams, hidden, ffn);
                    history = history is null ? b.Scale(summary, 0.5f) : b.Add(b.Scale(history, 0.5f), b.Scale(summary, 0.5f));
                    if (isAttention[il])
                    {
                        memory.Write(b, StaticStreamMixing.Mean(b, x, streams, hidden), Meta(b, summary, entropy, confidence, continuation));
                    }
                }

                using (b.Group($"passes.{pass}"))
                {
                    (x, var retrieved) = memory.Read(b, x);
                    var state = StaticStreamMixing.Mean(b, x, streams, hidden);
                    var diff = b.Sub(state, previous);
                    var features = DepthMemory.RouteFeatures(b, state, retrieved, diff, history);
                    continuation = DepthContinuationHead.Build(b, state, retrieved, diff, features, hidden, limit);
                    if (pass < passes - 1)
                    {
                        memory.Write(b, state, Meta(b, summary, entropy, confidence, continuation));
                    }

                    previous = state;
                }
            }

            using (b.Group("output"))
            {
                var hiddenStates = b.RmsNorm(StaticStreamMixing.Mean(b, x, streams, hidden), b.Weight("output_norm.weight"), eps, "result_norm");
                var rows = b.Gather(hiddenStates, outRows, "selected_rows");
                var head = b.TryWeight("lm_head.weight") ?? b.Weight("token_embedding.weight");
                b.Output("logits", b.MatMul(rows, head, "logits"));
                b.Output("hidden", hiddenStates);
            }

            return b.Build();
        }

        // Attention of physical layer p in pass `pass` (virtual layer v): fused q/k/v plus the pass's gated low-rank adapter, no positions.
        Value Attention(GraphBuilder b, Value input, Value positions, int pass, int v, string p)
        {
            var qkv = b.Linear(input, p + "attention.qkv.weight");
            var adapter = b.Linear(b.Linear(input, p + "attention.pass_lora.down.weight"), p + "attention.pass_lora.up.weight");
            qkv = b.Add(qkv, b.Mul(adapter, b.Slice(b.Weight(p + "attention.pass_gate"), 0, pass, 1)), $"{p}qkv_{pass}");
            int q = attention.Heads * attention.HeadDim;
            int kv = attention.KvHeads * attention.HeadDim;
            return attention.Attend(b, input, b.Slice(qkv, 1, 0, q), b.Slice(qkv, 1, q, kv), b.Slice(qkv, 1, q + kv, kv), positions, v, p,
                layerStates[v].First, layerStates[v].Second);
        }

        // The entry signals [s, entropy, confidence, continuation] (continuation 0 before the first pass ends).
        static Value Meta(GraphBuilder b, Value summary, Value entropy, Value confidence, Value? continuation)
            => b.Concat(1, summary, entropy, confidence, continuation ?? b.Scale(confidence, 0f));
    }

    private static WeightInfo Required(ModelSource src, string name)
        => src.Weight(name) ?? throw new InvalidDataException($"{src.Architecture}: missing weight '{name}'.");
}
