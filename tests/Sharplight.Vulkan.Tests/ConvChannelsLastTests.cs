using Sharplight.Cuda;
using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

/// <summary>
/// Reduced-precision convolutions over channels-last 16-bit activations (conv_cl): f16 operands (bf16 on devices with bf16 matrices
/// only, or with the f16 matrices turned off) and f32 accumulation against the f32 CPU oracle, with tolerances of the 16-bit
/// rounding of the operands (about 1e-2 of the output scale for f16, a few times that for bf16). Shapes cover partial tiles, groups,
/// strides, dilation, no padding, 1x1 and wide kernels, K splits and a symbolic batch; channel concatenations read only by the
/// convolution are staged straight into its input, with their f16-to-f32 casts and batch concatenations staged in place. Each case
/// also checks that the channels-last kernel ran where the device has the matrix features.
/// </summary>
public sealed class ConvChannelsLastTests
{
    private static Value Conv(TG tg, Value x, int c, int o, int k, int stride, int pad, int groups, int dilation = 1, string weightType = "f32")
    {
        var w = tg.RandW("w" + tg.Weights.Count, DataType.Get(weightType), 0.5f / MathF.Sqrt(c / groups * k * k), o, c / groups, k, k);
        var attrs = Attributes.From(("strides", new long[] { stride, stride }), ("pads", new long[] { pad, pad, pad, pad }),
            ("dilations", new long[] { dilation, dilation }), ("groups", (long)groups));
        return tg.B.Emit1("Conv", [x, w, tg.RandW("b" + tg.Weights.Count, DataType.F32, 0.5f, o)], attrs);
    }

    /// <summary>Runs <paramref name="g"/> against the oracle and asserts that a channels-last launch ran when the device supports it.</summary>
    private static void Check(string op, ComputeGraph g, IReadOnlyList<Step> steps, double atol, double rtol, string what, bool bf16 = false, int parts = 0,
        string? alsoLaunched = null)
    {
        ICompiledSegment? segment = null;
        IReadOnlySet<string>? disabled = bf16 ? new HashSet<string> { DeviceFeatures.MatrixF16 } : null;
        Diff.Check(op, g, steps, atol: atol, rtol: rtol, maxTokens: 2, allowReducedPrecision: true, disabledFeatures: disabled, what: what, inspect: m =>
        {
            segment = m.Segments[0];
            SetProfile(segment);
        });
        var features = Gpu.Device!.Info.Features;
        bool supported = bf16 ? features.Contains(DeviceFeatures.MatrixBF16) && features.Contains(DeviceFeatures.BFloat16)
            : features.Contains(DeviceFeatures.MatrixF16) && features.Contains(DeviceFeatures.Float16);
        if (supported && segment is not null && LaunchNames(segment) is { } names)
        {
            Assert.Contains(names, n => n.Contains(parts > 0 ? $"channels-last {parts} parts" : "channels-last", StringComparison.Ordinal));
            if (alsoLaunched is not null)
            {
                Assert.True(names.Any(n => n.Contains(alsoLaunched, StringComparison.Ordinal)), $"no {alsoLaunched} among: {string.Join("; ", names)}");
            }
        }
    }

    private static void SetProfile(ICompiledSegment segment)
    {
        switch (segment)
        {
            case VulkanSegment vulkan:
                vulkan.Profile = true;
                break;
            case CudaSegment cuda:
                cuda.Profile = true;
                break;
        }
    }

    private static IReadOnlyCollection<string>? LaunchNames(ICompiledSegment segment) => segment switch
    {
        VulkanSegment vulkan => vulkan.ProfileData.Keys,
        CudaSegment cuda => cuda.ProfileData.Keys,
        _ => null,
    };

    [Theory]
    [InlineData(64, 20, 96, 3, 1, 1, 1, 1, "f32")]     // partial channel and pixel tiles, split-K
    [InlineData(32, 33, 48, 3, 1, 1, 1, 1, "f16")]     // f16 weights, 48 outputs (one 64-row tile, partly empty)
    [InlineData(64, 17, 64, 3, 2, 1, 2, 1, "f32")]     // groups of 32 channels, stride 2
    [InlineData(128, 24, 256, 1, 1, 0, 1, 1, "f32")]   // 1x1, no padding, two 128-row tiles
    [InlineData(24, 30, 64, 5, 1, 2, 1, 1, "bf16")]    // 5x5, 24 channels (k groups cross taps within a step), bf16 weights
    [InlineData(64, 28, 128, 3, 1, 2, 1, 2, "f32")]    // dilation 2
    [InlineData(256, 7, 128, 3, 1, 1, 1, 1, "f32")]    // 7x7 map: 49 pixels, less than one tile
    [InlineData(64, 64, 64, 3, 1, 1, 1, 1, "f32")]     // 128-pixel tiles
    [InlineData(384, 40, 128, 3, 1, 0, 1, 1, "f32")]   // no padding, long K
    public void MatchesF32(int c, int hw, int o, int k, int stride, int pad, int groups, int dilation, string weightType)
    {
        var tg = new TG();
        var x = tg.InF("x", TG.T, c, hw, hw);
        tg.B.Output("y", Conv(tg, x, c, o, k, stride, pad, groups, dilation, weightType));
        int[] batches = [1, 2];
        var steps = batches.Select((n, i) => new Step(n, 0, (x, Rnd.F((long)n * c * hw * hw, 90 + i)))).ToArray();
        Check("Conv.channels-last", tg.B.Build(), steps, 2e-2, 1e-2, $"C={c} {hw}x{hw} O={o} k{k} s{stride} p{pad} g{groups} d{dilation} {weightType}");
    }

    /// <summary>bf16 operands (the form of devices without f16 matrices): 8 significant bits, so a looser tolerance.</summary>
    [Theory]
    [InlineData(64, 20, 96, 3)]
    [InlineData(128, 24, 256, 1)]
    public void BFloat16MatchesF32(int c, int hw, int o, int k)
    {
        var tg = new TG();
        var x = tg.InF("x", TG.T, c, hw, hw);
        tg.B.Output("y", Conv(tg, x, c, o, k, 1, k / 2, 1));
        Check("Conv.channels-last.bf16", tg.B.Build(), [new Step(1, 0, (x, Rnd.F((long)c * hw * hw, 95)))], 8e-2, 4e-2, $"C={c} {hw}x{hw} O={o} k{k}", bf16: true);
    }

    /// <summary>
    /// A convolution of a channel concatenation (a causal 3-D convolution's stacked frame windows): an f32 part, an f16 part cast to
    /// f32 and a batch concatenation of a cast f16 slice and an f32 slice; the result feeds a fused residual add and SiLU.
    /// </summary>
    [Fact]
    public void ConcatenatedInputWithEpilogue()
    {
        const int c = 32;
        const int hw = 18;
        var tg = new TG();
        var a = tg.In("a", DataType.F32, 2, c, hw, hw);
        var h = tg.In("h", DataType.F16, 3, c, hw, hw);
        var r = tg.In("r", DataType.F32, 2, 64, hw, hw);
        var cast = tg.B.Emit1("Cast", [tg.B.Slice(h, 0, 0, 2)], Attributes.From(("type", "f32")));
        var castOne = tg.B.Emit1("Cast", [tg.B.Slice(h, 0, 2, 1)], Attributes.From(("type", "f32")));
        var frames = tg.B.Concat(0, castOne, tg.B.Slice(a, 0, 1, 1));
        var x = tg.B.Concat(1, a, cast, frames);
        tg.B.Output("y", tg.B.Silu(tg.B.Add(Conv(tg, x, 3 * c, 64, 3, 1, 1, 1), r)));
        var half = Rnd.F(3L * c * hw * hw, 97).Select(v => (Half)v).ToArray();
        var step = new Step(1, 0, (a, Rnd.F(2L * c * hw * hw, 96)), (h, half), (r, Rnd.F(2L * 64 * hw * hw, 98)));
        Check("Conv.channels-last.concat", tg.B.Build(), [step], 2e-2, 1e-2, "concat");
    }

    /// <summary>
    /// A causal 3-D convolution over frames as the stacked windows of its pieces (as video VAEs build them): 3 time taps over 2 cached
    /// f16 frames and 5 f32 frames, pieces of 1 and 2 output frames whose windows are slices of the frames, casts of cached slices and
    /// batch concatenations of both. The convolution reads the windows in place (3 parts), the frames from one channels-last copy.
    /// </summary>
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public void StackedFrameWindows(int group)
    {
        const int c = 32;
        const int hw = 14;
        const int frames = 5;
        const int cached = 2;
        var tg = new TG();
        var x = tg.In("x", DataType.F32, frames, c, hw, hw);
        var cache = tg.In("cache", DataType.F16, cached, c, hw, hw);
        var w = tg.RandW("w", DataType.F32, 0.5f / MathF.Sqrt(3 * c * 9), 48, 3 * c, 3, 3);
        var attrs = Attributes.From(("strides", new long[] { 1, 1 }), ("pads", new long[] { 1, 1, 1, 1 }), ("dilations", new long[] { 1, 1 }), ("groups", 1L));
        Value Frames(long start, long count)
        {
            var parts = new List<Value>();
            if (start < cached)
            {
                long leading = Math.Min(start + count, cached) - start;
                parts.Add(tg.B.Emit1("Cast", [tg.B.Slice(cache, 0, start, leading)], Attributes.From(("type", "f32"))));
            }

            if (start + count > cached)
            {
                long from = Math.Max(start, cached) - cached;
                parts.Add(tg.B.Slice(x, 0, from, start + count - cached - from));
            }

            return parts.Count == 1 ? parts[0] : tg.B.Concat(0, [.. parts]);
        }

        var pieces = new List<Value>();
        for (int start = 0; start < frames; start += group)
        {
            int count = Math.Min(group, frames - start);
            var windows = Enumerable.Range(0, 3).Select(j => Frames(start + j, count)).ToArray();
            pieces.Add(tg.B.Emit1("Conv", [tg.B.Concat(1, windows), w, tg.RandW("b" + start, DataType.F32, 0.5f, 48)], attrs));
        }

        tg.B.Output("y", tg.B.Concat(0, [.. pieces]));
        var half = Rnd.F((long)cached * c * hw * hw, 101).Select(v => (Half)v).ToArray();
        var step = new Step(1, 0, (x, Rnd.F((long)frames * c * hw * hw, 100)), (cache, half));
        Check("Conv.channels-last.windows", tg.B.Build(), [step], 2e-2, 1e-2, $"group {group}", parts: 3, alsoLaunched: "copy conv_cl_stage");
    }

    /// <summary>
    /// Row bands of stacked frame windows (a causal convolution's pieces at large sizes): two overlapping bands of three frames, each
    /// window a band of one frame, read in place as views of the frames' channels-last copy at row offsets.
    /// </summary>
    [Fact]
    public void BandedFrameWindows()
    {
        const int c = 24;
        const int height = 20;
        const int width = 16;
        var tg = new TG();
        var x = tg.In("x", DataType.F32, 3, c, height, width);
        var w = tg.RandW("w", DataType.F32, 0.5f / MathF.Sqrt(3 * c * 9), 32, 3 * c, 3, 3);
        var attrs = Attributes.From(("strides", new long[] { 1, 1 }), ("pads", new long[] { 0, 1, 0, 1 }), ("dilations", new long[] { 1, 1 }), ("groups", 1L));
        var bands = new List<Value>();
        foreach (var (row, rows) in new[] { (0L, 12L), (8L, 12L) })
        {
            var windows = Enumerable.Range(0, 3).Select(j => tg.B.Slice(tg.B.Slice(x, 0, j, 1), 2, row, rows)).ToArray();
            bands.Add(tg.B.Emit1("Conv", [tg.B.Concat(1, windows), w], attrs));
        }

        tg.B.Output("y", tg.B.Concat(2, [.. bands]));
        var step = new Step(1, 0, (x, Rnd.F(3L * c * height * width, 102)));
        Check("Conv.channels-last.bands", tg.B.Build(), [step], 2e-2, 1e-2, "bands", parts: 3, alsoLaunched: "copy conv_cl_stage");
    }

    /// <summary>
    /// An f16 activation read directly (no conversion launch before the staging; the CPU oracle takes f32 convolution inputs, so
    /// it convolves the same values as f32).
    /// </summary>
    [Fact]
    public void HalfInput()
    {
        var data = Rnd.F(64L * 16 * 16, 99).Select(v => (Half)v).ToArray();
        ComputeGraph Build(DataType type, out Value x)
        {
            var tg = new TG();
            x = tg.In("x", type, TG.T, 64, 16, 16);
            tg.B.Output("y", Conv(tg, x, 64, 64, 3, 1, 1, 1));
            return tg.B.Build();
        }

        var half = Diff.RunVulkan(Build(DataType.F16, out var hx), [new Step(1, 0, (hx, data))], maxTokens: 2, allowReducedPrecision: true)[0];
        var oracle = Diff.RunCpu(Build(DataType.F32, out var fx), [new Step(1, 0, (fx, data.Select(v => (float)v).ToArray()))], maxTokens: 2)[0];
        for (int i = 0; i < oracle.Length; i++)
        {
            Assert.True(Math.Abs(half[i] - oracle[i]) <= 2e-2 + 1e-2 * Math.Abs(oracle[i]), $"element {i}: {half[i]} vs {oracle[i]}");
        }
    }
}
