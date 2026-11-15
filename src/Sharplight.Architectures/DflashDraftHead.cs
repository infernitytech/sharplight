using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// A DFlash block drafter (canonical architecture <c>dflash</c>, a separate checkpoint loaded as the <see cref="Vision.AuxiliaryRoles.Draft"/>
/// source of its target) as the target's <see cref="WellKnown.DraftGraph"/>. The target exports the inputs of several of its layers
/// (<c>dflash.target_layers</c>) as <see cref="TapsOutput"/>; the drafter fuses them, g = context_norm(fc(features)), and writes keys and values
/// of g into its KV cache at the target's positions. A draft pass then runs the pending token followed by <c>mask_token_id</c> tokens through
/// its decoder layers with bidirectional attention over the cache and the block, and every mask row predicts the token at its position.
/// </summary>
/// <remarks>
/// <para>
/// One pass serves both: the step rows of a sequence are its not yet written context rows (target features) followed by the block. The
/// <see cref="InputRole.HiddenStates"/> input is [T, F + 1 + 2·taps (+ 2)]: the target features F (zero on block rows), a context flag (1 on
/// context rows), per convolution tap j the step row j rows earlier and whether it lies in the same block, and with the selector the logits row
/// of the previous block row and whether that row is the pending token (<see cref="WellKnown.DraftBlockSize"/>). Every layer reads
/// a = n + flag·(g − n) in place of its normalized input n, so context rows store k_norm / rope(wk g) and wv g like llama.cpp's injection
/// pass, while their other results are unused; the caller cuts the block rows from the cache after the pass.
/// </para>
/// <para>
/// DFlash2 drafters add a dynamic depthwise convolution over the rows of the block around attention and MLP (coefficients
/// <c>conv.proj</c>(input) per group plus <c>conv.base</c>, input side before, output side after) and a candidate-lattice selector: the
/// <c>selector.top_k</c> best logits of every row with pairwise scores successorᵀ·(gate ⊙ predecessor) + logit from
/// <c>selector.{successor,predecessor,hidden}.weight</c>, output as the lattice rows of <see cref="WellKnown.DraftLatticeCandidates"/>. Without
/// the selector the logits output is the greedy token of every row (one column). The sliding window of a windowed layer counts from the last
/// row of the step's run (the block's last position), which shortens the window of earlier block rows by up to the block length.
/// </para>
/// <para>
/// Hyper-parameters (<c>dflash.*</c>): <c>hidden_size</c>, <c>layer_count</c>, <c>head_count</c>, <c>kv_head_count</c>, <c>head_dim</c>,
/// <c>ffn_hidden_size</c>, <c>norm_eps</c>, <c>rope.base</c>, <c>rope.dims</c>, optional <c>rope.attention_scale</c>, <c>target_layers</c>,
/// <c>block_size</c>, <c>mask_token_id</c>, <c>causal</c>, optional <c>sliding_window</c> with <c>sliding_layers</c> (int[] 0/1),
/// <c>conv.kernel_size</c> / <c>conv.group_size</c>, <c>selector.top_k</c> / <c>selector.rank</c>, <c>embedding_scale</c>,
/// <c>attention.scale</c>, <c>attention.value_scale</c>, <c>logit_scale</c>, <c>final_logit_softcap</c>, <c>mlp.activation</c> ("silu",
/// "gelu_tanh"). Weights: <c>fc.weight</c> [H, n·H_target], <c>context_norm.weight</c>, per layer <c>input_norm.weight</c>,
/// <c>attention.{q,k,v,o}.weight</c>, <c>attention.{q_norm,k_norm}.weight</c>, optional <c>attention.sinks</c>,
/// <c>attention.post_norm.weight</c>, <c>post_attention_norm.weight</c>, <c>mlp.{gate,up,down}.weight</c>, optional
/// <c>mlp.post_norm.weight</c>, <c>output_scale</c>, <c>{attention,mlp}.conv.base</c> [2, K, H] and <c>{attention,mlp}.conv.proj.weight</c>
/// [2·K·H/group, H]; <c>output_norm.weight</c>, optional <c>lm_head.weight</c>, <c>token_embedding.weight</c> (default the target's),
/// <c>rope.freq_factors</c>, <c>draft_to_target</c>, <c>selector.*.weight</c>.
/// </para>
/// </remarks>
public static class DflashDraftHead
{
    /// <summary>Canonical architecture id of DFlash drafter sources.</summary>
    public const string Architecture = "dflash";

    /// <summary>The decode-graph output the drafter reads: the concatenated layer inputs of the target.</summary>
    public const string TapsOutput = Eagle3DraftHead.TapsOutput;

    /// <summary>Prefix of the drafter's weights in the target's definition.</summary>
    public const string WeightPrefix = "draft.";

    /// <summary>Whether <paramref name="source"/> is a DFlash drafter.</summary>
    public static bool IsDrafter(ModelSource? source) => source is not null && source.Architecture == Architecture;

    /// <summary>The target layers whose inputs the drafter fuses (<c>dflash.target_layers</c>).</summary>
    /// <exception cref="InvalidDataException">The drafter names no layers.</exception>
    public static int[] TargetLayers(ModelSource drafter)
    {
        var layers = drafter.Arch("target_layers", Array.Empty<int>());
        return layers.Length > 0 ? layers : throw new InvalidDataException("dflash: target_layers is empty.");
    }

    /// <summary>Earlier block rows the drafter's convolution reads (kernel size − 1; 0 without one).</summary>
    public static int ConvTaps(ModelSource drafter) => Math.Max(0, drafter.Arch("conv.kernel_size", 0) - 1);

    /// <summary>Candidates per row of the drafter's selector lattice (0 without one).</summary>
    public static int LatticeCandidates(ModelSource drafter) => drafter.Arch("selector.top_k", 0);

    /// <summary>The draft properties of the target's definition (<see cref="WellKnown.DraftBlockSize"/> and the ones it refers to).</summary>
    /// <exception cref="InvalidDataException">The drafter has no mask token.</exception>
    public static Dictionary<string, object> Properties(ModelSource drafter)
    {
        var props = new Dictionary<string, object>
        {
            [WellKnown.DraftLayers] = drafter.Arch<int>("layer_count"),
            [WellKnown.DraftHiddenSource] = TapsOutput,
            [WellKnown.DraftBlockSize] = Math.Max(2, drafter.Arch("block_size", 16)),
            [WellKnown.DraftMaskToken] = drafter.Arch("mask_token_id", -1) is >= 0 and var mask ? mask
                : throw new InvalidDataException("dflash: the drafter has no mask_token_id."),
            [WellKnown.DraftConvTaps] = ConvTaps(drafter),
            [WellKnown.DraftLatticeCandidates] = LatticeCandidates(drafter),
        };
        if (Eagle3DraftHead.TokenMap(drafter) is { } map)
        {
            props[WellKnown.DraftTokenMap] = map;
        }

        return props;
    }

    /// <summary>The drafter's states: a KV cache per layer.</summary>
    /// <param name="drafter">The drafter source.</param>
    /// <param name="options">Adapter options of the target (KV cache type).</param>
    public static IReadOnlyList<StateDefinition> States(ModelSource drafter, AdapterOptions options)
    {
        int kvHeads = drafter.Arch<int>("kv_head_count");
        int headDim = drafter.Arch<int>("head_dim");
        var shape = new Shape(Dim.Sym(WellKnown.ContextSymbol), kvHeads, headDim);
        var type = options.KvCacheTypeFor(headDim);
        var states = new List<StateDefinition>();
        for (int il = 0; il < drafter.Arch<int>("layer_count"); il++)
        {
            states.Add(new StateDefinition { Name = $"dflash.{il}.k_cache", Type = type, Shape = shape, Kind = StateKind.KeyCache, Group = "dflash" });
            states.Add(new StateDefinition { Name = $"dflash.{il}.v_cache", Type = type, Shape = shape, Kind = StateKind.ValueCache, Group = "dflash" });
        }

        return states;
    }

    /// <summary>The drafter's weights under <see cref="WeightPrefix"/>, to merge into the target's weights.</summary>
    public static IEnumerable<KeyValuePair<string, WeightInfo>> PrefixedWeights(ModelSource drafter) => Eagle3DraftHead.PrefixedWeights(drafter);

    /// <summary>Builds the <see cref="WellKnown.DraftGraph"/>.</summary>
    /// <param name="drafter">The drafter source.</param>
    /// <param name="weights">
    /// Weight lookup of the target with the drafter's weights merged in under <see cref="WeightPrefix"/>; the target's token embedding and LM head
    /// serve a drafter without its own.
    /// </param>
    /// <param name="targetHidden">The target's hidden size.</param>
    /// <param name="positionAxes">Position axes of the target's decode graph (only the first is used).</param>
    /// <param name="states">The drafter's states (<see cref="States"/>).</param>
    /// <exception cref="InvalidDataException">The drafter's hyper-parameters do not fit the target.</exception>
    public static ComputeGraph Build(ModelSource drafter, Func<string, WeightInfo?> weights, int targetHidden, int positionAxes,
        IReadOnlyList<StateDefinition> states)
    {
        var p = new DflashParameters(drafter, targetHidden);
        var b = new GraphBuilder(WellKnown.DraftGraph, weights);
        var T = Dim.Sym(WellKnown.TokenSymbol);
        var tokens = b.Input("tokens", InputRole.TokenIds, DataType.I32, new Shape(T));
        var hiddenIn = b.Input("hidden_states", InputRole.HiddenStates, DataType.F32, new Shape(T, p.InputWidth));
        var positions = b.Input("positions", InputRole.Positions, DataType.I32, new Shape(T, positionAxes));
        var outRows = b.Input("output_rows", InputRole.OutputRows, DataType.I32, new Shape(Dim.Sym("R")));
        var pos = positionAxes == 1 ? positions : b.Slice(positions, 1, 0, 1);
        Value x, g, contextFlag;
        Value[] tapRows = new Value[p.ConvTaps];
        Value[] tapValid = new Value[p.ConvTaps];
        using (b.Group("input"))
        {
            var features = b.Slice(hiddenIn, 1, 0, p.Features);
            g = b.RmsNorm(b.Linear(features, D("fc.weight"), name: "dflash_fc"), b.Weight(D("context_norm.weight")), p.Eps, "dflash_context");
            contextFlag = b.Slice(hiddenIn, 1, p.Features, 1);
            for (int j = 0; j < p.ConvTaps; j++)
            {
                tapRows[j] = Index(b, b.Slice(hiddenIn, 1, p.Features + 1 + (2 * j), 1));
                tapValid[j] = b.Slice(hiddenIn, 1, p.Features + 2 + (2 * j), 1);
            }

            var table = b.TryWeight(D("token_embedding.weight")) ?? b.Weight("token_embedding.weight");
            x = b.Gather(table, tokens, "dflash_embd");
            if (p.EmbeddingScale != 0f)
            {
                x = b.Scale(x, p.EmbeddingScale);
            }
        }

        var rope = Attributes.From(("dims", (long)p.RopeDims), ("base", p.RopeBase), ("style", "neox"));
        if (p.RopeAttentionScale != 1f)
        {
            rope = rope.With("attn_factor", p.RopeAttentionScale);
        }

        var factors = b.TryWeight(D("rope.freq_factors"));
        Value Rotate(Value t) => b.Emit1("Rope", factors is null ? [t, pos] : [t, pos, factors], rope);
        b.RuntimeScalar(WellKnown.PastScalar);
        for (int il = 0; il < p.Layers; il++)
        {
            using var _ = b.Group($"dflash.layers.{il}");
            string l = $"layers.{il}.";
            var n = b.RmsNorm(x, b.Weight(D(l + "input_norm.weight")), p.Eps, $"dflash_noise_norm_{il}");
            Value? attnCoefficients = null;
            if (p.ConvTaps > 0)
            {
                attnCoefficients = b.Linear(n, D(l + "attention.conv.proj.weight"), name: $"dflash_attn_conv_proj_{il}");
                n = Convolve(b, p, n, attnCoefficients, D(l + "attention.conv.base"), 0, tapRows, tapValid);
            }

            // Context rows: the fused target features; block rows: the normalized (convolved) noise.
            var a = b.Add(n, b.Mul(contextFlag, b.Binary("Sub", g, n)), $"dflash_attn_input_{il}");
            var q = Rotate(b.RmsNorm(b.Reshape(b.Linear(a, D(l + "attention.q.weight"), name: $"dflash_q_{il}"), 0, p.Heads, p.HeadDim),
                b.Weight(D(l + "attention.q_norm.weight")), p.Eps));
            var k = Rotate(b.RmsNorm(b.Reshape(b.Linear(a, D(l + "attention.k.weight"), name: $"dflash_k_{il}"), 0, p.KvHeads, p.HeadDim),
                b.Weight(D(l + "attention.k_norm.weight")), p.Eps));
            var v = b.Reshape(b.Linear(a, D(l + "attention.v.weight"), name: $"dflash_v_{il}"), 0, p.KvHeads, p.HeadDim);
            var kState = b.State(states[2 * il]);
            var vState = b.State(states[(2 * il) + 1]);
            b.CacheWrite(kState, k);
            b.CacheWrite(vState, v);
            var attrs = Attributes.From(("scale", p.AttentionScale), ("offset", WellKnown.PastScalar), ("causal", p.Causal));
            if (p.Window > 0 && p.SlidingLayers[il])
            {
                attrs = attrs.With("window", p.Window);
            }

            Value[] attnIn = [q, kState, vState];
            if (b.TryWeight(D(l + "attention.sinks")) is { } sinks)
            {
                attnIn = [q, kState, vState, sinks];
                attrs = attrs.With("sinks", true);
            }

            var attended = b.Emit1("CachedAttention", attnIn, attrs, $"dflash_attn_{il}");
            var o = b.Linear(b.Reshape(attended, 0, (long)p.Heads * p.HeadDim), D(l + "attention.o.weight"), name: $"dflash_attn_out_{il}");
            if (p.ValueScale != 0f)
            {
                o = b.Scale(o, p.ValueScale);
            }

            if (attnCoefficients is not null)
            {
                o = Convolve(b, p, o, attnCoefficients, D(l + "attention.conv.base"), 1, tapRows, tapValid);
            }

            if (b.TryWeight(D(l + "attention.post_norm.weight")) is { } attnPost)
            {
                o = b.RmsNorm(o, attnPost, p.Eps);
            }

            var h = b.Add(o, x, $"dflash_ffn_input_{il}");
            var f = b.RmsNorm(h, b.Weight(D(l + "post_attention_norm.weight")), p.Eps, $"dflash_ffn_norm_{il}");
            Value? mlpCoefficients = null;
            if (p.ConvTaps > 0)
            {
                mlpCoefficients = b.Linear(f, D(l + "mlp.conv.proj.weight"), name: $"dflash_mlp_conv_proj_{il}");
                f = Convolve(b, p, f, mlpCoefficients, D(l + "mlp.conv.base"), 0, tapRows, tapValid);
            }

            var gate = b.Linear(f, D(l + "mlp.gate.weight"));
            var up = b.Linear(f, D(l + "mlp.up.weight"));
            var m = b.Linear(b.Mul(b.Unary(p.Activation, gate), up), D(l + "mlp.down.weight"), name: $"dflash_ffn_out_{il}");
            if (mlpCoefficients is not null)
            {
                m = Convolve(b, p, m, mlpCoefficients, D(l + "mlp.conv.base"), 1, tapRows, tapValid);
            }

            if (b.TryWeight(D(l + "mlp.post_norm.weight")) is { } mlpPost)
            {
                m = b.RmsNorm(m, mlpPost, p.Eps);
            }

            x = b.Add(m, h, $"dflash_out_{il}");
            if (b.TryWeight(D(l + "output_scale")) is { } outScale)
            {
                x = b.Mul(x, outScale);
            }
        }

        using (b.Group("output"))
        {
            var normed = b.RmsNorm(x, b.Weight(D("output_norm.weight")), p.Eps, "dflash_result_norm");
            var rows = b.Gather(normed, outRows, "selected_rows");
            var head = b.TryWeight(D("lm_head.weight")) ?? b.TryWeight("lm_head.weight") ?? b.Weight("token_embedding.weight");
            var logits = b.MatMul(rows, head, "dflash_logits");
            if (p.LatticeCandidates > 0)
            {
                b.Output(WellKnown.LogitsOutput, Lattice(b, p, logits, rows, tokens, hiddenIn, outRows));
            }
            else
            {
                b.Output(WellKnown.LogitsOutput, b.Emit1("Cast", [b.Emit1("ArgMax", [logits])], Attributes.From(("type", "f32")), "dflash_greedy"));
            }
        }

        return b.Build();
    }

    private static string D(string name) => WeightPrefix + name;

    /// <summary>An i32 row index [T] from an f32 column [T, 1].</summary>
    private static Value Index(GraphBuilder b, Value column)
        => b.Emit1("Cast", [b.Reshape(column, 0)], Attributes.From(("type", "i32")));

    /// <summary>
    /// The DFlash2 dynamic depthwise convolution over the rows of a block: out[t, c] = Σ_tap (coefficients[t, side, tap, c / group] +
    /// base[side, tap, c]) · x[t − tap, c], rows before the block's first row reading zero.
    /// </summary>
    private static Value Convolve(GraphBuilder b, DflashParameters p, Value x, Value coefficients, string baseWeight, int side,
        Value[] tapRows, Value[] tapValid)
    {
        int groups = p.Hidden / p.ConvGroupSize;
        var bases = b.Weight(baseWeight);   // [2, K, H]
        Value? result = null;
        for (int tap = 0; tap <= p.ConvTaps; tap++)
        {
            var c = b.Slice(coefficients, 1, ((side * (p.ConvTaps + 1)) + tap) * groups, groups);   // [T, G]
            var perChannel = b.Reshape(b.Emit1("Tile", [b.Reshape(c, 0, groups, 1)], Attributes.From(("repeats", new long[] { 1, 1, p.ConvGroupSize }))), 0, p.Hidden);
            var weight = b.Add(perChannel, b.Reshape(b.Slice(b.Slice(bases, 0, side, 1), 1, tap, 1), p.Hidden));
            var values = tap == 0 ? x : b.Mul(b.Gather(x, tapRows[tap - 1]), tapValid[tap - 1]);
            var term = b.Mul(weight, values);
            result = result is null ? term : b.Add(result, term);
        }

        return result!;
    }

    /// <summary>
    /// The DFlash2 candidate lattice of the logits rows: [R, k + k·k] with the k best tokens of the row and score[p·k + c] =
    /// successor[cand c] · (gate ⊙ predecessor[cand p of the previous row]) + logit of c (the previous row being the pending token: its token).
    /// </summary>
    private static Value Lattice(GraphBuilder b, DflashParameters p, Value logits, Value rows, Value tokens, Value hiddenIn, Value outRows)
    {
        int k = p.LatticeCandidates;
        int rank = p.SelectorRank;
        if (p.LogitScale != 0f)
        {
            logits = b.Scale(logits, p.LogitScale);
        }

        if (p.Softcap > 0f)
        {
            logits = b.Scale(b.Unary("Tanh", b.Scale(logits, 1f / p.Softcap)), p.Softcap);
        }

        var top = b.Emit("TopK", [logits], Attributes.From(("k", (long)k)));
        var unary = top[0];        // [R, k]
        var candidates = top[1];   // [R, k] i32
        var gate = b.Linear(rows, D("selector.hidden.weight"), name: "dflash_selector_gate");   // [R, rank]
        var successorTable = b.Weight(D("selector.successor.weight"));       // [V, rank]
        var predecessorTable = b.Weight(D("selector.predecessor.weight"));   // [V, rank]
        var successors = new Value[k];
        var predecessors = new Value[k];
        for (int c = 0; c < k; c++)
        {
            var ids = b.Reshape(b.Slice(candidates, 1, c, 1), 0);
            successors[c] = b.Gather(successorTable, ids);
            predecessors[c] = b.Gather(predecessorTable, ids);
        }

        var successor = b.Mul(b.Reshape(b.Concat(1, successors), 0, k, rank), b.Reshape(gate, 0, 1, rank));    // [R, k, rank]
        var ownCandidates = b.Reshape(b.Concat(1, predecessors), 0, (long)k * rank);                             // [R, k·rank]
        var ownToken = b.Gather(b.Gather(predecessorTable, tokens), outRows);                                     // [R, rank]

        // The previous block row of every logits row (host-provided), and whether that row is the pending token.
        // The two link columns end the input row; the gather reads the last 8 columns (a row width every device gathers), then keeps them.
        var link = b.Slice(b.Gather(b.Slice(hiddenIn, 1, p.InputWidth - 8, 8), outRows), 1, 6, 2);               // [R, 2]
        var previous = Index(b, b.Slice(link, 1, 0, 1));
        var fromToken = b.Reshape(b.Slice(link, 1, 1, 1), 0, 1, 1);
        var fromCandidates = b.Reshape(b.Gather(ownCandidates, previous), 0, k, rank);
        var anchor = b.Reshape(b.Gather(ownToken, previous), 0, 1, rank);
        var predecessor = b.Add(fromCandidates, b.Mul(fromToken, b.Binary("Sub", anchor, fromCandidates)));    // [R, k, rank]
        var scores = b.Emit1("BatchMatMul", [predecessor, successor], Attributes.From(("trans_b", true)), "dflash_lattice_scores");   // [R, k(p), k(c)]
        scores = b.Add(scores, b.Reshape(unary, 0, 1, k));
        return b.Concat(1, b.Emit1("Cast", [candidates], Attributes.From(("type", "f32"))), b.Reshape(scores, 0, (long)k * k));
    }
}
