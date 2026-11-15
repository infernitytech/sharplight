using Sharplight.Graph;

namespace Sharplight.Vulkan.Tests;

/// <summary>Hashed n-gram row ids (<c>NGramHash</c>), Vulkan against the CPU oracle: missing tokens (also the token itself) and stop-token cuts.</summary>
public sealed class NGramHashTests
{
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void NGramHashMatchesCpu(bool stopCuts)
    {
        Gpu.Require();
        const int Order = 3;
        const int Rows = 40;
        var tg = new TG();
        var window = tg.InF("window", TG.T, Order);
        long[] multipliers = [1000003, 7919, 104729];
        long[] sizes = [31, 37, 41, 43];
        long[] offsets = [0, 31, 68, 109];
        tg.B.Output("ids", tg.B.Emit1("NGramHash", [window], Attributes.From(("multipliers", multipliers), ("head_sizes", sizes), ("head_offsets", offsets),
            ("heads_per_order", 2L), ("stop_token", 5L), ("stop_cuts", stopCuts))));

        // Tokens + 1 in 0..9: 0 is a missing token, 6 the stop token.
        var random = new Random(3);
        var data = Enumerable.Range(0, Rows * Order).Select(_ => (float)random.Next(10)).ToArray();
        Diff.Check("NGramHash", tg.B.Build(), [new Step(Rows, 0, (window, data))], maxTokens: Rows, what: $"stop_cuts={stopCuts}");
    }
}
