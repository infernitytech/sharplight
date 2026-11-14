using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// An EAGLE-3 drafter (canonical architecture <c>eagle3</c>, a separate checkpoint loaded as the <see cref="Vision.AuxiliaryRoles.Draft"/> source
/// of its target) as the target's <see cref="WellKnown.DraftGraph"/>. The target exports the inputs of three of its layers
/// (<c>eagle3.target_layers</c>) as <see cref="TapsOutput"/>; the drafter fuses them into its own hidden size, g = fc(features), and runs one
/// decoder layer whose attention reads concat(input_norm(embed(x_{i+1})), hidden_norm(g_i)) and whose residual is g (or its normalized form with
/// <c>eagle3.norm_before_residual</c>); the output norm and LM head predict the token after x_{i+1}. Chained drafts feed back the layer's output
/// before the output norm in place of g.
/// </summary>
/// <remarks>
/// <para>
/// The graph's <see cref="InputRole.HiddenStates"/> input is [T, 3·H_target + H]: rows built from the target carry its features in the first part
/// and zeros in the second, chained rows zeros and the fed-back state; since fc has no bias, g = fc(features) + fed-back state serves both.
/// </para>
/// <para>
/// Hyper-parameters (<c>eagle3.*</c>): <c>hidden_size</c>, <c>head_count</c>, <c>kv_head_count</c>, <c>head_dim</c>, <c>norm_eps</c>,
/// <c>rope.base</c>, <c>rope.style</c> ("neox" default, "gptj"), <c>target_layers</c> (int[3]: the target layers whose inputs are fused; the
/// target's layer count means its last layer's output), <c>target_hidden_size</c>, <c>norm_before_residual</c>, <c>norm_before_fc</c>.
/// Weights: <c>fc.weight</c> [H, 3·H_target], optional <c>fc_norm.weight</c> [3·H_target] (an RMS norm of the features before fc),
/// <c>layers.0.{input_norm,hidden_norm,post_attention_norm}.weight</c>, <c>layers.0.attention.{q,k,v}.weight</c> [·, 2H],
/// <c>layers.0.attention.o.weight</c>, <c>layers.0.mlp.{gate,up,down}.weight</c>, <c>output_norm.weight</c>, optional <c>lm_head.weight</c>
/// [V_draft, H] (default the target's head), optional <c>token_embedding.weight</c> (default the target's), optional <c>rope.freq_factors</c> and
/// optional <c>draft_to_target</c> (integer [V_draft]: the target token of every draft token, <see cref="WellKnown.DraftTokenMap"/>).
/// </para>
/// </remarks>
public static class Eagle3DraftHead
{
    /// <summary>Canonical architecture id of EAGLE-3 drafter sources.</summary>
    public const string Architecture = "eagle3";

    /// <summary>The decode-graph output the drafter reads: the concatenated layer inputs of the target, [T, 3·H_target].</summary>
    public const string TapsOutput = "hidden_taps";

    /// <summary>Prefix of the drafter's weights in the target's definition.</summary>
    public const string WeightPrefix = "draft.";

    private const string Layer = "layers.0.";

    /// <summary>Whether <paramref name="source"/> is an EAGLE-3 drafter.</summary>
    public static bool IsDrafter(ModelSource? source) => source is not null && source.Architecture == Architecture;

    /// <summary>The target layers whose inputs the drafter fuses (<c>eagle3.target_layers</c>).</summary>
    /// <exception cref="InvalidDataException">The drafter does not name three layers.</exception>
    public static int[] TargetLayers(ModelSource drafter)
    {
        var layers = drafter.Arch("target_layers", Array.Empty<int>());
        return layers.Length == 3 ? layers : throw new InvalidDataException("eagle3: target_layers must name three target layers.");
    }

    /// <summary>The drafter's state: one KV cache.</summary>
    /// <param name="drafter">The drafter source.</param>
    /// <param name="options">Adapter options of the target (KV cache type).</param>
    public static IReadOnlyList<StateDefinition> States(ModelSource drafter, AdapterOptions options)
    {
        int kvHeads = drafter.Arch<int>("kv_head_count");
        int headDim = drafter.Arch<int>("head_dim");
        var shape = new Shape(Dim.Sym(WellKnown.ContextSymbol), kvHeads, headDim);
        var type = options.KvCacheTypeFor(headDim);
        return
        [
            new StateDefinition { Name = "eagle3.k_cache", Type = type, Shape = shape, Kind = StateKind.KeyCache, Group = "eagle3" },
            new StateDefinition { Name = "eagle3.v_cache", Type = type, Shape = shape, Kind = StateKind.ValueCache, Group = "eagle3" },
        ];
    }

    /// <summary>The drafter's weights under <see cref="WeightPrefix"/>, to merge into the target's weights.</summary>
    public static IEnumerable<KeyValuePair<string, WeightInfo>> PrefixedWeights(ModelSource drafter)
        => drafter.Weights.Select(kv => KeyValuePair.Create(WeightPrefix + kv.Key, kv.Value.As(WeightPrefix + kv.Key)));

    /// <summary>The target token of every draft token (<c>draft_to_target</c>), or null when both vocabularies are the same.</summary>
    /// <exception cref="InvalidDataException">The map is not an integer vector.</exception>
    public static int[]? TokenMap(ModelSource drafter)
    {
        if (drafter.Weight("draft_to_target") is not { } map)
        {
            return null;
        }

        var bytes = map.ReadAll();
        int n = (int)map.Shape.Elements();
        var ids = new int[n];
        for (int i = 0; i < n; i++)
        {
            ids[i] = map.Type == DataType.I64 ? checked((int)BitConverter.ToInt64(bytes, i * 8))
                : map.Type == DataType.I32 ? BitConverter.ToInt32(bytes, i * 4)
                : throw new InvalidDataException($"eagle3: draft_to_target must be i32 or i64, not {map.Type}.");
        }

        return ids;
    }

    /// <summary>Builds the <see cref="WellKnown.DraftGraph"/>.</summary>
    /// <param name="drafter">The drafter source.</param>
    /// <param name="weights">
    /// Weight lookup of the target with the drafter's weights merged in under <see cref="WeightPrefix"/> (<see cref="PrefixedWeights"/>); the
    /// target's token embedding and LM head serve a drafter without its own.
    /// </param>
    /// <param name="targetHidden">The target's hidden size.</param>
    /// <param name="positionAxes">Position axes of the target's decode graph (only the first is used).</param>
    /// <param name="states">The drafter's states (<see cref="States"/>).</param>
    /// <exception cref="InvalidDataException">The drafter's hyper-parameters do not fit the target.</exception>
    public static ComputeGraph Build(ModelSource drafter, Func<string, WeightInfo?> weights, int targetHidden, int positionAxes,
        IReadOnlyList<StateDefinition> states)
    {
        int hidden = drafter.Arch<int>("hidden_size");
        int heads = drafter.Arch<int>("head_count");
        int kvHeads = drafter.Arch<int>("kv_head_count");
        int headDim = drafter.Arch<int>("head_dim");
        float eps = drafter.Arch("norm_eps", 1e-5f);
        int features = 3 * drafter.Arch("target_hidden_size", targetHidden);
        if (features != 3 * targetHidden)
        {
            throw new InvalidDataException($"eagle3: the drafter fuses layers of width {features / 3}, the target's are {targetHidden}.");
        }

        static string D(string name) => WeightPrefix + name;

        var b = new GraphBuilder(WellKnown.DraftGraph, weights);
        var T = Dim.Sym(WellKnown.TokenSymbol);
        Value x, residual, attnInput;
        var tokens = b.Input("tokens", InputRole.TokenIds, DataType.I32, new Shape(T));
        var hiddenIn = b.Input("hidden_states", InputRole.HiddenStates, DataType.F32, new Shape(T, features + hidden));
        var positions = b.Input("positions", InputRole.Positions, DataType.I32, new Shape(T, positionAxes));
        var outRows = b.Input("output_rows", InputRole.OutputRows, DataType.I32, new Shape(Dim.Sym("R")));
        using (b.Group("input"))
        {
            var fused = b.Slice(hiddenIn, 1, 0, features);
            if (b.TryWeight(D("fc_norm.weight")) is { } fcNorm)
            {
                fused = b.RmsNorm(fused, fcNorm, eps, "eagle3_fc_norm");
            }

            var g = b.Add(b.Linear(fused, D("fc.weight"), name: "eagle3_fc"), b.Slice(hiddenIn, 1, features, hidden), "eagle3_g");
            var table = b.TryWeight(D("token_embedding.weight")) ?? b.Weight("token_embedding.weight");
            var e = b.RmsNorm(b.Gather(table, tokens, "eagle3_embd"), b.Weight(D(Layer + "input_norm.weight")), eps, "eagle3_embd_norm");
            var gNorm = b.RmsNorm(g, b.Weight(D(Layer + "hidden_norm.weight")), eps, "eagle3_g_norm");
            residual = drafter.Arch("norm_before_residual", false) ? gNorm : g;
            attnInput = b.Concat(1, e, gNorm);
        }

        using (b.Group("eagle3.layers.0"))
        {
            var pos = positionAxes == 1 ? positions : b.Slice(positions, 1, 0, 1);
            var factors = b.TryWeight(D("rope.freq_factors"));
            var ropeAttributes = Attributes.From(("dims", (long)headDim), ("base", drafter.Arch("rope.base", 10000f)), ("style", drafter.Arch("rope.style", "neox")));
            Value Rotate(Value t) => b.Emit1("Rope", factors is null ? [t, pos] : [t, pos, factors], ropeAttributes);
            var q = Rotate(b.Reshape(b.Linear(attnInput, D(Layer + "attention.q.weight"), name: "eagle3_q"), 0, heads, headDim));
            var k = Rotate(b.Reshape(b.Linear(attnInput, D(Layer + "attention.k.weight"), name: "eagle3_k"), 0, kvHeads, headDim));
            var v = b.Reshape(b.Linear(attnInput, D(Layer + "attention.v.weight"), name: "eagle3_v"), 0, kvHeads, headDim);
            var kState = b.State(states[0]);
            var vState = b.State(states[1]);
            b.CacheWrite(kState, k);
            b.CacheWrite(vState, v);
            b.RuntimeScalar(WellKnown.PastScalar);
            var attended = b.Emit1("CachedAttention", [q, kState, vState],
                Attributes.From(("scale", 1f / MathF.Sqrt(headDim)), ("offset", WellKnown.PastScalar)), "eagle3_attn");
            var o = b.Linear(b.Reshape(attended, 0, (long)heads * headDim), D(Layer + "attention.o.weight"), name: "eagle3_attn_out");
            x = b.Add(o, residual, "eagle3_attn_residual");
            var h = b.RmsNorm(x, b.Weight(D(Layer + "post_attention_norm.weight")), eps, "eagle3_ffn_norm");
            var gate = b.Linear(h, D(Layer + "mlp.gate.weight"));
            var up = b.Linear(h, D(Layer + "mlp.up.weight"));
            x = b.Add(b.Linear(b.Mul(b.Unary("Silu", gate), up), D(Layer + "mlp.down.weight"), name: "eagle3_ffn_out"), x, "eagle3_out");
        }

        using (b.Group("output"))
        {
            var normed = b.RmsNorm(x, b.Weight(D("output_norm.weight")), eps, "eagle3_result_norm");
            var rows = b.Gather(normed, outRows, "selected_rows");
            var head = b.TryWeight(D("lm_head.weight")) ?? b.TryWeight("lm_head.weight") ?? b.Weight("token_embedding.weight");
            b.Output("logits", b.MatMul(rows, head, "logits"));
            b.Output(WellKnown.HiddenOutput, x);
        }

        return b.Build();
    }
}
