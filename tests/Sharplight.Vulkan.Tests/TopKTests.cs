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
