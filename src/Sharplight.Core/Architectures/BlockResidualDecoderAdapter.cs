using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Decoder with block attention residuals and a looped core (canonical arch <c>block_residual_decoder</c>; Inkling 2 Mini): every layer is a
/// <see cref="GatedSinkAttention"/> mixer and a routed feed-forward block, each followed by a residual short convolution, and reads its inputs as
/// attention-weighted mixes of the residual stream's banked block checkpoints (<see cref="AttentionResiduals"/> over normalized rows). An optional
/// core of layers runs several times; every pass after the first restarts from the core's input and gates messages from the previous pass into
/// its layers' inputs.
/// </summary>
/// <remarks>
/// Layer i with input x (the running partial sum of its block):
/// <list type="bullet">
/// <item>Attention inputs: layer 0 reads x; a sliding layer one mix <c>a = mix(x; layers.i.attention_res.score.weight[0])</c>, a full layer one per
/// query, key and value (rows 0, 1, 2). With messages, each input passes its message gate first. Sliding layers normalize them all with
/// <c>input_norm.weight</c>, full layers with <c>{q,k,v}_input_norm.weight</c>.</item>
/// <item><c>y = conv(attention)</c> (<c>post_mixer_conv.weight</c>, see <see cref="GatedSinkAttention.ResidualConv"/>). A layer whose index is a
/// multiple of <c>attention_residuals.block_size</c> banks x as a checkpoint and restarts the partial sum at y; the others continue it,
/// r = x + y.</item>
/// <item><c>m = mix(r; layers.i.mlp_res.score.weight)</c> (gated with the MLP message), then the output <c>r + conv(moe(post_attention_norm(m)))</c>
/// (<c>mlp.out_conv.weight</c>; routed experts per <c>moe.*</c> and <c>mlp.activation</c>, see <see cref="MoeBlock"/>).</item>
/// </list>
/// Loop (<c>loop.first_layer</c> F, <c>loop.end_layer</c> E, <c>loop.passes</c> P &gt; 1): layers [F, E) run P times with KV caches and
/// convolution windows of their own per pass; pass n &gt; 0 starts again from the hidden rows and checkpoints the core started with and, for
/// loop layer j, mixes four messages (query, key, value, MLP) over the checkpoints the previous pass banked after its first one plus its output
/// (the queries <c>loop.messages.query_mlp.weight</c> [L, 2, H] and <c>loop.messages.key_value.weight</c> [L, 2, H]). A message gate
/// (<c>layers.i.message_gate.{q,k,v,mlp}.{down,forget,gain}.weight</c>) mixes message m into input x as <c>(1 − tanh(F·d))·x + tanh(G·d)·m</c>
/// with <c>d = D·[rms(x), rms(m)]</c> (weight-less RMS norms).
/// Output: <c>mix(x; output_res.score.weight)</c>, <c>output_norm.weight</c>, <c>lm_head.weight</c>, the soft cap <c>logits.softcap</c> and the
/// suppressed ranges <c>logits.suppressed_ranges</c>. The embeddings are RMS-normalized with <c>embedding_norm.weight</c>.
/// Hyper-parameters (<c>block_residual_decoder.*</c>): <c>layer_count</c>, <c>hidden_size</c>, <c>norm_eps</c>, <c>context_length</c>,
/// <c>attention_residuals.{block_size,normalized_values,score_scale}</c>, <c>loop.*</c>, <c>logits.*</c>, <c>mlp.*</c>, <c>moe.*</c> and the
/// attention keys of <see cref="GatedSinkAttention"/>.
/// </remarks>
public sealed class BlockResidualDecoderAdapter : IModelAdapter
{
    /// <inheritdoc/>
    public string Name => "Block-residual decoder (attention residuals, gated sink attention, looped core)";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => ["block_residual_decoder"];

    private static readonly string[] s_messageGates = ["q", "k", "v", "mlp"];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        int layerCount = src.Arch<int>("layer_count");
        int hidden = src.Arch<int>("hidden_size");
        float eps = src.Arch("norm_eps", 1e-6f);
        float softcap = src.Arch("logits.softcap", 0f);
        int[] suppressedRanges = src.Arch("logits.suppressed_ranges", Array.Empty<int>());
        int blockSize = src.Arch("attention_residuals.block_size", 0);
        bool normalizedValues = src.Arch("attention_residuals.normalized_values", false);
        float scoreScale = src.Arch("attention_residuals.score_scale", 1f);
        int loopFirst = src.Arch("loop.first_layer", 0);
        int loopEnd = src.Arch("loop.end_layer", 0);
        int passes = Math.Max(1, src.Arch("loop.passes", 1));
        if (blockSize <= 0)
        {
            throw new InvalidDataException("block_residual_decoder: attention_residuals.block_size must be positive.");
        }

        if (passes > 1 && (loopFirst <= 0 || loopEnd <= loopFirst || loopEnd > layerCount || loopFirst % blockSize != 0))
        {
            throw new InvalidDataException("block_residual_decoder: the loop must be a range [first, end) of layers starting at a block boundary after layer 0.");
        }

        var attention = new GatedSinkAttention(src);
        var activation = MlpActivations.Create(src.Arch("mlp.activation", "silu"), src);
        var moe = MoeOptions.Read(src, normalizeDefault: true);

        // States per layer and pass: KV caches, the key / value convolution windows, the attention and MLP output convolution windows.
        var states = new List<StateDefinition>();
        var layerStates = new IReadOnlyList<StateDefinition>[layerCount, passes];
        for (int il = 0; il < layerCount; il++)
        {
            string group = $"layers.{il}";
            int layerPasses = passes > 1 && il >= loopFirst && il < loopEnd ? passes : 1;
            for (int pass = 0; pass < layerPasses; pass++)
            {
                string names = pass == 0 ? "" : $"pass{pass}.";
                IReadOnlyList<StateDefinition> layer =
                [
                    .. attention.DefineStates(src, group, pass, il, options.KvCacheType),
                    OutputConvState(src, group, names + "mixer_conv_state", $"layers.{il}.post_mixer_conv.weight"),
                    OutputConvState(src, group, names + "ffn_conv_state", $"layers.{il}.mlp.out_conv.weight"),
                ];
                layerStates[il, pass] = layer;
                states.AddRange(layer);
            }
        }

        var graphs = new Dictionary<string, ComputeGraph> { [WellKnown.DecodeGraph] = BuildTrunk(WellKnown.DecodeGraph, false) };
        if (options.IncludeAuxiliaryGraphs)
        {
            graphs["decode_embeddings"] = BuildTrunk("decode_embeddings", true);
        }

        return RecurrentModelDefinition.Create(src, layerCount, hidden, graphs, states);

        ComputeGraph BuildTrunk(string name, bool embeddingsInput)
        {
            var b = new GraphBuilder(name, src.Weight);
            var constants = new ConstantCache(b);
            var residuals = new AttentionResiduals(blockSize, eps, normalizedValues, scoreScale);
            var T = Dim.Sym(WellKnown.TokenSymbol);
            Value x;
            using (b.Group("input"))
            {
                if (embeddingsInput)
                {
                    x = b.Input("embeddings", InputRole.Embeddings, DataType.F32, new Shape(T, hidden));
                }
                else
                {
                    x = b.Gather(b.Weight("token_embedding.weight"), b.Input("tokens", InputRole.TokenIds, DataType.I32, new Shape(T)), "inp_embd");
                    if (b.TryWeight("embedding_norm.weight") is { } embeddingNorm)
                    {
                        x = b.RmsNorm(x, embeddingNorm, eps, "inp_embd_norm");
                    }
                }
            }

            var positions = b.Input("positions", InputRole.Positions, DataType.I32, new Shape(T, 1));
            var outRows = b.Input("output_rows", InputRole.OutputRows, DataType.I32, new Shape(Dim.Sym("R")));
            int coreFirst = passes > 1 ? loopFirst : layerCount;
            int coreEnd = passes > 1 ? loopEnd : layerCount;
            for (int il = 0; il < coreFirst; il++)
            {
                x = LayerGroup(b, x, positions, il, 0, null, residuals, constants);
            }

            if (passes > 1)
            {
                var coreInput = x;
                var coreBank = residuals.Save();
                for (int il = coreFirst; il < coreEnd; il++)
                {
                    x = LayerGroup(b, x, positions, il, 0, null, residuals, constants);
                }

                for (int pass = 1; pass < passes; pass++)
                {
                    // The previous pass's checkpoints after its first (the core input, the same in every pass) and its output.
                    var previous = residuals.Since(b, coreBank.Count + 1);
                    var previousOutput = x;
                    residuals.Restore(coreBank);
                    x = coreInput;
                    for (int il = coreFirst; il < coreEnd; il++)
                    {
                        Value[] messages;
                        using (b.Group($"layers.{il}"))
                        {
                            messages = Messages(b, previous, previousOutput, il - coreFirst, $"layers.{il}.pass{pass}");
                        }

                        x = LayerGroup(b, x, positions, il, pass, messages, residuals, constants);
                    }
                }

                for (int il = coreEnd; il < layerCount; il++)
                {
                    x = LayerGroup(b, x, positions, il, 0, null, residuals, constants);
                }
            }

            using (b.Group("output"))
            {
                var mixed = residuals.Mix(b, x, "output_res.score.weight", hidden, "output_mix");
                var hiddenStates = b.RmsNorm(mixed, b.Weight("output_norm.weight"), eps, "result_norm");
                var rows = b.Gather(hiddenStates, outRows, "selected_rows");
                var head = b.TryWeight("lm_head.weight") ?? b.Weight("token_embedding.weight");
                var logits = b.MatMul(rows, head, "logits");
                if (softcap > 0f)
                {
                    logits = b.Scale(b.Unary("Tanh", b.Scale(logits, 1f / softcap)), softcap);
                }

                b.Output("logits", SuppressedLogits.Apply(b, logits, checked((int)head.Shape[0].Fixed), suppressedRanges));
                b.Output("hidden", hiddenStates);
            }

            return b.Build();
        }

        Value LayerGroup(GraphBuilder b, Value x, Value positions, int il, int pass, Value[]? messages, AttentionResiduals residuals, ConstantCache constants)
        {
            using var _ = b.Group($"layers.{il}");
            return Layer(b, x, positions, il, pass, messages, residuals, constants);
        }

        // The four messages [T, H] (query, key, value, MLP) of loop layer j from the previous pass's checkpoints and output.
        Value[] Messages(GraphBuilder b, AttentionResiduals previous, Value previousOutput, int j, string name)
        {
            var queryMlp = b.Reshape(b.Slice(b.Weight("loop.messages.query_mlp.weight"), 0, j, 1), 2, hidden);
            var keyValue = b.Reshape(b.Slice(b.Weight("loop.messages.key_value.weight"), 0, j, 1), 2, hidden);
            Value Row(Value queries, int row) => b.Reshape(b.Slice(queries, 0, row, 1), hidden);
            return
            [
                previous.Mix(b, previousOutput, Row(queryMlp, 0), hidden, $"{name}.q_message"),
                previous.Mix(b, previousOutput, Row(keyValue, 0), hidden, $"{name}.k_message"),
                previous.Mix(b, previousOutput, Row(keyValue, 1), hidden, $"{name}.v_message"),
                previous.Mix(b, previousOutput, Row(queryMlp, 1), hidden, $"{name}.mlp_message"),
            ];
        }

        Value Layer(GraphBuilder b, Value x, Value positions, int il, int pass, Value[]? messages, AttentionResiduals residuals, ConstantCache constants)
        {
            string p = $"layers.{il}.";
            var layer = layerStates[il, pass];
            bool sliding = attention.Sliding(il);
            Value hq, hk, hv;
            if (sliding)
            {
                var a = il == 0 ? x : residuals.Mix(b, x, ScoreRow(b, p, 0), hidden, $"{p}attn_mix");
                var norm = b.Weight(p + "input_norm.weight");
                hq = b.RmsNorm(Gated(b, a, messages, 0, p), norm, eps);
                hk = messages is null ? hq : b.RmsNorm(Gated(b, a, messages, 1, p), norm, eps);
                hv = messages is null ? hq : b.RmsNorm(Gated(b, a, messages, 2, p), norm, eps);
            }
            else
            {
                Value Input(int row) => Gated(b, il == 0 ? x : residuals.Mix(b, x, ScoreRow(b, p, row), hidden, $"{p}attn_mix{row}"), messages, row, p);
                hq = b.RmsNorm(Input(0), b.Weight(p + "q_input_norm.weight"), eps);
                hk = b.RmsNorm(Input(1), b.Weight(p + "k_input_norm.weight"), eps);
                hv = b.RmsNorm(Input(2), b.Weight(p + "v_input_norm.weight"), eps);
            }

            var mixed = GatedSinkAttention.ResidualConv(b, attention.Build(b, hq, hk, hv, positions, il, layer, constants), p + "post_mixer_conv.weight", layer[4]);
            Value residual;
            if (il % blockSize == 0)
            {
                residuals.Push(b, x, hidden);
                residual = mixed;
            }
            else
            {
                residual = b.Add(x, mixed, $"{p}attn_residual");
            }

            var m = Gated(b, residuals.Mix(b, residual, p + "mlp_res.score.weight", hidden, $"{p}mlp_mix"), messages, 3, p);
            var ffn = MoeBlock.Build(b, b.RmsNorm(m, b.Weight(p + "post_attention_norm.weight"), eps, $"{p}post_attn_norm"), p, "mlp.", moe, hidden, activation);
            ffn = GatedSinkAttention.ResidualConv(b, ffn, p + "mlp.out_conv.weight", layer[5]);
            return b.Add(residual, ffn, $"{p}l_out");
        }

        // Row r of the layer's attention-input scores [n, H] as a vector [H].
        Value ScoreRow(GraphBuilder b, string p, int row) => b.Reshape(b.Slice(b.Weight(p + "attention_res.score.weight"), 0, row, 1), hidden);

        // The message gate of input r (q, k, v, mlp) of a loop layer; the input itself without messages.
        Value Gated(GraphBuilder b, Value x, Value[]? messages, int r, string p)
        {
            if (messages is null)
            {
                return x;
            }

            string g = $"{p}message_gate.{s_messageGates[r]}.";
            var message = messages[r];
            var d = b.Linear(b.Concat(1, b.RmsNorm(x, null, eps), b.RmsNorm(message, null, eps)), g + "down.weight");
            var forget = b.Unary("Tanh", b.Linear(d, g + "forget.weight"));
            var gain = b.Unary("Tanh", b.Linear(d, g + "gain.weight"));
            return b.Add(b.Sub(x, b.Mul(forget, x)), b.Mul(gain, message), $"{g}out");
        }

        static StateDefinition OutputConvState(ModelSource src, string group, string name, string weightName)
        {
            var w = src.Weight(weightName) ?? throw new InvalidDataException($"block_residual_decoder: missing {weightName}.");
            return RecurrentStates.ConvWindow(group, name, (int)w.Shape[1].Fixed - 1, (int)w.Shape[0].Fixed);
        }
    }
}
