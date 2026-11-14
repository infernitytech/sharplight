using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Builds the <see cref="WellKnown.DraftGraph"/> of a model with multi-token-prediction (MTP / NextN) layers, the part every
/// family shares: row <c>i</c> combines the embedding of the token after position <c>i</c> with the target's hidden state at
/// <c>i</c>, <c>fc(concat(norm_e(embed(x_{i+1})), norm_h(h_i)))</c>, runs the family's decoder layers (supplied by the adapter, with
/// their own KV caches), and predicts the token two positions ahead through the output norm and the (shared) LM head.
/// </summary>
/// <remarks>
/// Canonical weights: <c>mtp.fc.weight</c> [H, 2H] (input order embedding | hidden), <c>mtp.embedding_norm.weight</c>,
/// <c>mtp.hidden_norm.weight</c>, <c>mtp.output_norm.weight</c>, optional <c>mtp.token_embedding.weight</c> /
/// <c>mtp.lm_head.weight</c> (default: the trunk's <c>token_embedding.weight</c> and <c>lm_head.weight</c>, else the tied
/// embedding) and the layers' weights below <c>mtp.layers.{j}.</c>.
/// <para>
/// The target's hidden state is its <see cref="WellKnown.HiddenOutput"/> (after the final norm), and the graph's own
/// <see cref="WellKnown.HiddenOutput"/>, which the next chained draft step feeds back, is taken after the output norm too
/// (<c>chainNormalized</c>): Qwen3.5 / Qwen3-Next and the NextN heads of DeepSeek-V3 / GLM-4.5 alike (vLLM, SGLang and llama.cpp
/// feed these heads the normed states).
/// </para>
/// </remarks>
public static class MtpDraftHead
{
    private const string FirstModule = "mtp.layers.0.";
    private static readonly string[] s_moduleWeights = ["fc.weight", "embedding_norm.weight", "hidden_norm.weight", "output_norm.weight"];

    /// <summary>True when <paramref name="lookup"/> has the shared MTP weights (fc and the two input norms).</summary>
    public static bool IsPresent(Func<string, WeightInfo?> lookup)
        => lookup("mtp.fc.weight") is not null && lookup("mtp.embedding_norm.weight") is not null && lookup("mtp.hidden_norm.weight") is not null;

    /// <summary>
    /// The NextN layout (DeepSeek-V3, GLM-4.5): every MTP module <c>j</c> is one decoder layer with its own fc, input norms and
    /// optional output norm, all below <c>mtp.layers.{j}.</c> (<c>fc.weight</c>, <c>embedding_norm.weight</c>,
    /// <c>hidden_norm.weight</c>, <c>output_norm.weight</c>, else the trunk's <c>output_norm.weight</c>). A single module is a complete
    /// draft step (chained for further drafts, as vLLM and llama.cpp do); this returns a lookup that serves module 0 under the shared
    /// names <see cref="Build"/> reads, for a head of one layer (weights below <c>mtp.layers.0.</c>), or null when the model has no
    /// such module. The embedding and LM head stay the trunk's: the modules share them with the main model (the copies a checkpoint
    /// carries per module are not used, so no second copy is loaded).
    /// </summary>
    /// <param name="lookup">Weight lookup of the model.</param>
    public static Func<string, WeightInfo?>? FirstNextnModule(Func<string, WeightInfo?> lookup)
    {
        if (s_moduleWeights.Take(3).Any(n => lookup(FirstModule + n) is null))
        {
            return null;
        }

        return name =>
        {
            if (name is "mtp.token_embedding.weight" or "mtp.lm_head.weight")
            {
                return null;
            }

            string suffix = name.StartsWith("mtp.", StringComparison.Ordinal) ? name["mtp.".Length..] : "";
            if (!s_moduleWeights.Contains(suffix))
            {
                return lookup(name);
            }

            return lookup(FirstModule + suffix) ?? (suffix == "output_norm.weight" ? lookup("output_norm.weight") : null);
        };
    }

    /// <summary>Model properties of a NextN head (<see cref="FirstNextnModule"/>): one draft layer; empty without a head.</summary>
    /// <param name="present">Whether the model has the head.</param>
    public static IReadOnlyDictionary<string, object> NextnProperties(bool present) => present
        ? new Dictionary<string, object> { [WellKnown.DraftLayers] = 1 }
        : new Dictionary<string, object>();

    /// <summary>Builds the draft graph.</summary>
    /// <param name="lookup">Weight lookup of the model.</param>
    /// <param name="hiddenSize">Hidden size H (of the target's hidden states and of the MTP layers).</param>
    /// <param name="positionAxes">Position axes of the decode graph (1, or 4 for M-RoPE).</param>
    /// <param name="eps">RMS norm epsilon.</param>
    /// <param name="layers">Number of MTP layers.</param>
    /// <param name="layer">Builds MTP layer <c>j</c> (weights below <c>mtp.layers.{j}.</c>) on <c>x</c> with the positions input and returns its output.</param>
    /// <param name="chainNormalized">Whether the <see cref="WellKnown.HiddenOutput"/> is taken after the output norm.</param>
    public static ComputeGraph Build(Func<string, WeightInfo?> lookup, int hiddenSize, int positionAxes, float eps, int layers,
        Func<GraphBuilder, Value, int, Value, Value> layer, bool chainNormalized)
    {
        var b = new GraphBuilder(WellKnown.DraftGraph, lookup);
        var T = Dim.Sym(WellKnown.TokenSymbol);
        Value x;
        using (b.Group("input"))
        {
            var tokens = b.Input("tokens", InputRole.TokenIds, DataType.I32, new Shape(T));
            var hidden = b.Input("hidden_states", InputRole.HiddenStates, DataType.F32, new Shape(T, hiddenSize));
            var table = b.TryWeight("mtp.token_embedding.weight") ?? b.Weight("token_embedding.weight");
            var e = b.RmsNorm(b.Gather(table, tokens, "mtp_embd"), b.Weight("mtp.embedding_norm.weight"), eps, "mtp_embd_norm");
            var h = b.RmsNorm(hidden, b.Weight("mtp.hidden_norm.weight"), eps, "mtp_hidden_norm");
            x = b.Linear(b.Concat(1, e, h), "mtp.fc.weight", name: "mtp_fc");
        }

        var positions = b.Input("positions", InputRole.Positions, DataType.I32, new Shape(T, positionAxes));
        var outRows = b.Input("output_rows", InputRole.OutputRows, DataType.I32, new Shape(Dim.Sym("R")));
        for (int j = 0; j < layers; j++)
        {
            using var _ = b.Group($"mtp.layers.{j}");
            x = layer(b, x, j, positions);
        }

        using (b.Group("output"))
        {
            var normed = b.RmsNorm(x, b.Weight("mtp.output_norm.weight"), eps, "mtp_result_norm");
            var rows = b.Gather(normed, outRows, "selected_rows");
            var head = b.TryWeight("mtp.lm_head.weight") ?? b.TryWeight("lm_head.weight") ?? b.Weight("token_embedding.weight");
            b.Output("logits", b.MatMul(rows, head, "logits"));
            b.Output(WellKnown.HiddenOutput, chainNormalized ? normed : x);
        }

        return b.Build();
    }
}
