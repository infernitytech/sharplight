using Sharplight.Graph;
using Sharplight.Runtime;

namespace Sharplight.Vulkan.Tests;

/// <summary>
/// The parallel top-k kernels against the CPU oracle: the subgroup kernel (small k), the per-row workgroup kernel
/// (radix select + bitonic sort; also forced for small k by disabling subgroup operations) and the serial fallback
/// for k beyond the shared-memory sort.
/// </summary>
public sealed class TopKTests
{
    private static readonly IReadOnlySet<string> s_noSubgroups = new HashSet<string> { DeviceFeatures.SubgroupOps };

    [Theory]
    [InlineData(8400, 300, true, 1, false)]    // detection head: workgroup kernel
    [InlineData(24000, 300, true, 1, false)]
    [InlineData(1000, 1000, false, 1, false)]  // k == n: full sort
    [InlineData(37, 5, true, 1, false)]        // small k: subgroup kernel
    [InlineData(128, 8, true, 1, false)]       // MoE router shape
    [InlineData(128, 8, true, 1, true)]        // ... on the workgroup kernel
    [InlineData(300, 16, false, 1, true)]
    [InlineData(4100, 3000, true, 1, false)]   // k beyond the shared-memory sort: serial fallback
    public void Rows(int n, int k, bool largest, int seed, bool disableSubgroups)
    {
        var tg = new TG();
        var x = tg.InF("x", TG.T, n);
        var outs = tg.B.Emit("TopK", [x], Attributes.From(("k", (long)k), ("axis", 1L), ("largest", largest)));
        tg.B.Output("v", outs[0]);
        tg.B.Output("i", outs[1]);
        Diff.Check("TopK.rows", tg.B.Build(), new Step(3, 0, (x, Rnd.F(3L * n, seed))), atol: 0, rtol: 0,
            what: $"n={n} k={k} largest={largest} noSubgroups={disableSubgroups}", disabledFeatures: disableSubgroups ? s_noSubgroups : null);
    }

    /// <summary>
    /// A mixture-of-experts router, softmax then top-k, with and without the renormalization of the selected weights (sum, + epsilon,
    /// divide): one fused launch on the GPU (with subgroups) or the separate kernels, against the CPU oracle. The ids match exactly, the
    /// weights to float rounding.
    /// </summary>
    [Theory]
    [InlineData(256, 8, true, false)]    // Qwen3.5-122B: 256 experts, top 8, renormalized
    [InlineData(128, 8, false, false)]
    [InlineData(512, 10, true, false)]   // Qwen3-Coder-Next: 512 experts, top 10 (the row in registers)
    [InlineData(1024, 8, false, false)]  // 32 values per lane
    [InlineData(100, 6, true, false)]    // not a multiple of the subgroup: per-lane lists
    [InlineData(64, 4, true, true)]      // without subgroups: the unfused kernels
    public void Router(int experts, int k, bool normalize, bool disableSubgroups)
    {
        var tg = new TG();
        var logits = tg.InF("logits", TG.T, experts);
        var top = tg.B.Emit("TopK", [tg.B.Emit1("Softmax", [logits])], Attributes.From(("k", (long)k)));
        var weights = top[0];
        if (normalize)
        {
            weights = tg.B.Binary("Div", weights, tg.B.Scale(tg.B.Emit1("ReduceSum", [weights]), 1f, 1e-20f));
        }

        tg.B.Output("w", weights);
        tg.B.Output("i", top[1]);
        Diff.Check("TopK.router", tg.B.Build(), new Step(5, 0, (logits, Rnd.F(5L * experts, 11, 4f))), atol: 1e-6, rtol: 1e-5,
            what: $"experts={experts} k={k} normalize={normalize} noSubgroups={disableSubgroups}", disabledFeatures: disableSubgroups ? s_noSubgroups : null);
    }

    /// <summary>
    /// Routers over logits with heavy ties (a few levels, so equal probabilities): the selected experts resolve ties to the lower
    /// index as the CPU oracle does, in the fused softmax-then-top-k launch and in the top-k-then-softmax one (gpt-oss).
    /// </summary>
    [Theory]
    [InlineData(512, 10, false)]
    [InlineData(128, 8, false)]
    [InlineData(128, 4, true)]
    [InlineData(32, 4, true)]
    public void RouterTies(int experts, int k, bool selectedSoftmax)
    {
        var tg = new TG();
        var logits = tg.InF("logits", TG.T, experts);
        Value weights;
        Value ids;
        if (selectedSoftmax)
        {
            var top = tg.B.Emit("TopK", [logits], Attributes.From(("k", (long)k)));
            weights = tg.B.Emit1("Softmax", [top[0]]);
            ids = top[1];
        }
        else
        {
            var top = tg.B.Emit("TopK", [tg.B.Emit1("Softmax", [logits])], Attributes.From(("k", (long)k)));
            weights = top[0];
            ids = top[1];
        }

        tg.B.Output("w", weights);
        tg.B.Output("i", ids);
        var data = Rnd.F(4L * experts, 13, 4f).Select(v => MathF.Round(v * 2) / 2).ToArray();
        Diff.Check("TopK.routerTies", tg.B.Build(), new Step(4, 0, (logits, data)), atol: 1e-6, rtol: 1e-5,
            what: $"experts={experts} k={k} selectedSoftmax={selectedSoftmax}");
    }

    /// <summary>Heavy ties (values quantized to a few levels) and signed zeros: ties must resolve to the lower index.</summary>
    [Theory]
    [InlineData(2000, 300, true, false)]
    [InlineData(2000, 300, false, false)]
    [InlineData(500, 12, true, false)]
    [InlineData(500, 12, false, true)]
    public void Ties(int n, int k, bool largest, bool disableSubgroups)
    {
        var tg = new TG();
        var x = tg.InF("x", TG.T, n);
        var outs = tg.B.Emit("TopK", [x], Attributes.From(("k", (long)k), ("axis", -1L), ("largest", largest)));
        tg.B.Output("v", outs[0]);
        tg.B.Output("i", outs[1]);
        var data = Rnd.F(2L * n, 7).Select(v => MathF.Round(v * 3) / 3).ToArray();
        for (int i = 0; i < data.Length; i += 17)
        {
            data[i] = i % 34 == 0 ? -0f : 0f;
        }

        Diff.Check("TopK.ties", tg.B.Build(), new Step(2, 0, (x, data)), atol: 0, rtol: 0,
            what: $"n={n} k={k} largest={largest} noSubgroups={disableSubgroups}", disabledFeatures: disableSubgroups ? s_noSubgroups : null);
    }

    /// <summary>Top-k along a middle axis (strided rows).</summary>
    [Theory]
    [InlineData(4, false)]
    [InlineData(40, false)]
    [InlineData(40, true)]
    public void MiddleAxis(int k, bool disableSubgroups)
    {
        var tg = new TG();
        var x = tg.InF("x", TG.T, 300, 6);
        var outs = tg.B.Emit("TopK", [x], Attributes.From(("k", (long)k), ("axis", 1L), ("largest", true)));
        tg.B.Output("v", outs[0]);
        tg.B.Output("i", outs[1]);
        Diff.Check("TopK.axis", tg.B.Build(), new Step(2, 0, (x, Rnd.F(2 * 300 * 6, 11))), atol: 0, rtol: 0,
            what: $"k={k} noSubgroups={disableSubgroups}", disabledFeatures: disableSubgroups ? s_noSubgroups : null);
    }

    /// <summary>
    /// A symbolic k along the symbolic outermost axis (<c>min(K, T)</c>, the detection post-processing pattern): the workgroup kernel,
    /// and the serial kernel for k bounds beyond its shared-memory sort.
    /// </summary>
    [Theory]
    [InlineData(200, 1, 700, 3)]       // k = 200 of 700
    [InlineData(200, 1, 150, 4)]       // k = all 150 rows
    [InlineData(50, 3, 400, 5)]        // three columns
    [InlineData(9000, 1, 2500, 6)]     // serial kernel
    public void SymbolicK(long kMax, int columns, int rows, int seed)
    {
        var tg = new TG();
        var x = tg.InF("x", TG.T, columns);
        var outs = tg.B.Emit("TopK", [x], Attributes.From(("k_symbol", $"min({kMax},T)"), ("axis", 0L), ("largest", true)));
        tg.B.Output("v", outs[0]);
        tg.B.Output("i", outs[1]);
        var data = Rnd.F((long)rows * columns, seed).Select(v => MathF.Round(v * 50) / 50).ToArray();
        Diff.Check("TopK.symbolic", tg.B.Build(), new Step(rows, 0, (x, data)), atol: 0, rtol: 0, maxTokens: 3000, what: $"k=min({kMax},T) columns={columns} T={rows}");
    }
}
