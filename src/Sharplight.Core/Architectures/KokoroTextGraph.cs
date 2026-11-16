using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// The first Kokoro graph over the phoneme symbols (symbolic length <c>T</c>): the ALBERT text encoder, the duration encoder (bidirectional LSTMs with adaptive
/// layer norms conditioned on the predictor style), the duration head and the convolutional text encoder. Outputs the summed duration sigmoids
/// (<see cref="DurationOutput"/>, <c>[T, 1]</c>, frames per symbol before rounding), the prosody features (<see cref="ProsodyOutput"/>, <c>[T, 640]</c>) and the
/// acoustic features (<see cref="AcousticOutput"/>, <c>[T, 512]</c>); the host expands the last two by the rounded durations.
/// </summary>
internal static class KokoroTextGraph
{
    /// <summary>The graph name.</summary>
    public const string GraphName = "kokoro_text";

    /// <summary>The input of the symbol ids, i32 <c>[T]</c>.</summary>
    public const string IdsInput = "ids";

    /// <summary>The input of the positions <c>0 .. T - 1</c>, i32 <c>[T]</c>.</summary>
    public const string PositionsInput = "positions";

    /// <summary>The input of the predictor style, f32 <c>[1, style]</c>.</summary>
    public const string StyleInput = "style";

    /// <summary>The output of the summed duration sigmoids, f32 <c>[T, 1]</c>.</summary>
    public const string DurationOutput = "duration";

    /// <summary>The output of the prosody features, f32 <c>[T, hidden + style]</c>.</summary>
    public const string ProsodyOutput = "prosody";

    /// <summary>The output of the acoustic features, f32 <c>[T, hidden]</c>.</summary>
    public const string AcousticOutput = "acoustic";

    /// <summary>Builds the graph.</summary>
    /// <param name="p">The hyper-parameters.</param>
    /// <param name="weights">Finds a weight by name.</param>
    public static ComputeGraph Build(KokoroParameters p, Func<string, WeightInfo?> weights)
    {
        var b = new GraphBuilder(GraphName, weights);
        var ids = b.Input(IdsInput, InputRole.TokenIds, DataType.I32, new Shape(Dim.Sym(WellKnown.TokenSymbol)));
        var positions = b.Input(PositionsInput, InputRole.Positions, DataType.I32, new Shape(Dim.Sym(WellKnown.TokenSymbol)));
        var style = b.Input(StyleInput, InputRole.Other, DataType.F32, new Shape(1, p.StyleDim));

        var bert = Albert(b, p, ids, positions);
        var dEn = b.Linear(bert, "bert_encoder.weight", "bert_encoder.bias");                                             // [T, hidden]
        var d = DurationEncoder(b, p, dEn, style);                                                                         // [T, hidden + style]
        var rnn = KokoroBlocks.BiLstm(b, d, "predictor.lstm");
        var logits = b.Linear(rnn, "predictor.duration_proj.linear_layer.weight", "predictor.duration_proj.linear_layer.bias");
        var duration = b.Emit1("Reduce", [b.Sigmoid(logits)], Attributes.From(("mode", "sum"), ("axes", new long[] { 1 }), ("keepdims", true)));

        b.Output(DurationOutput, duration);
        b.Output(ProsodyOutput, d);
        b.Output(AcousticOutput, TextEncoder(b, p, ids));
        return b.Build();
    }

    /// <summary>The ALBERT encoder: embeddings, projection, one shared transformer layer applied <see cref="KokoroParameters.BertLayers"/> times.</summary>
    private static Value Albert(GraphBuilder b, KokoroParameters p, Value ids, Value positions)
    {
        var words = b.Gather(b.Weight("bert.embeddings.word_embeddings.weight"), ids);
        var places = b.Gather(b.Weight("bert.embeddings.position_embeddings.weight"), positions);
        var types = b.Slice(b.Weight("bert.embeddings.token_type_embeddings.weight"), 0, 0, 1);
        var x = b.Add(b.Add(words, places), types);
        x = LayerNorm(b, x, "bert.embeddings.LayerNorm", 1e-12f);
        x = b.Linear(x, "bert.encoder.embedding_hidden_mapping_in.weight", "bert.encoder.embedding_hidden_mapping_in.bias");
        const string layer = "bert.encoder.albert_layer_groups.0.albert_layers.0.";
        long heads = p.BertHeads;
        long headDim = p.BertHidden / heads;
        for (int il = 0; il < p.BertLayers; il++)
        {
            using var _ = b.Group($"bert.layer.{il}");
            Value Project(string name) => b.Reshape(b.Linear(x, layer + "attention." + name + ".weight", layer + "attention." + name + ".bias"), 0, heads, headDim);
            var attended = b.Emit1("Attention", [Project("query"), Project("key"), Project("value")], Attributes.From(("scale", 1f / MathF.Sqrt(headDim)), ("causal", false)));
            var o = b.Linear(b.Reshape(attended, 0, p.BertHidden), layer + "attention.dense.weight", layer + "attention.dense.bias");
            var h = LayerNorm(b, b.Add(x, o), layer + "attention.LayerNorm", 1e-12f);
            var ff = b.Linear(b.Unary("GeluTanh", b.Linear(h, layer + "ffn.weight", layer + "ffn.bias")), layer + "ffn_output.weight", layer + "ffn_output.bias");
            x = LayerNorm(b, b.Add(h, ff), layer + "full_layer_layer_norm", 1e-12f);
        }

        return x;
    }

    /// <summary>The duration encoder: alternating bidirectional LSTMs and adaptive layer norms, the style concatenated before every LSTM.</summary>
    private static Value DurationEncoder(GraphBuilder b, KokoroParameters p, Value features, Value style)
    {
        var x = b.Concat(1, features, KokoroBlocks.StyleRows(b, style, features));
        for (int il = 0; il < p.Layers; il++)
        {
            using var _ = b.Group($"predictor.text_encoder.{il}");
            x = KokoroBlocks.BiLstm(b, x, $"predictor.text_encoder.lstms.{2 * il}");
            x = KokoroBlocks.AdaLayerNorm(b, x, style, $"predictor.text_encoder.lstms.{(2 * il) + 1}", p.Hidden);
            x = b.Concat(1, x, KokoroBlocks.StyleRows(b, style, x));
        }

        return x;
    }

    /// <summary>The acoustic text encoder: embedding, convolution / layer norm / leaky ReLU blocks, bidirectional LSTM.</summary>
    private static Value TextEncoder(GraphBuilder b, KokoroParameters p, Value ids)
    {
        var x = b.Gather(b.Weight("text_encoder.embedding.weight"), ids);
        for (int il = 0; il < p.Layers; il++)
        {
            using var _ = b.Group($"text_encoder.cnn.{il}");
            x = KokoroBlocks.Conv(b, x, $"text_encoder.cnn.{il}.0", (p.TextKernel - 1) / 2);
            x = b.Emit1("LayerNorm", [x, b.Weight($"text_encoder.cnn.{il}.1.gamma"), b.Weight($"text_encoder.cnn.{il}.1.beta")], Attributes.From(("eps", 1e-5f)));
            x = KokoroBlocks.LeakyRelu(b, x, 0.2f);
        }

        return KokoroBlocks.BiLstm(b, x, "text_encoder.lstm");
    }

    private static Value LayerNorm(GraphBuilder b, Value x, string name, float eps)
        => b.Emit1("LayerNorm", [x, b.Weight(name + ".weight"), b.Weight(name + ".bias")], Attributes.From(("eps", eps)));
}
