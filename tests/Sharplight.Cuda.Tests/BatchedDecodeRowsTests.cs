using System.Text;
using System.Text.RegularExpressions;
using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;
using Sharplight.Vulkan.Tests;

namespace Sharplight.Cuda.Tests;

/// <summary>
/// Batched decode rows give each row's single-row result on the CUDA provider (docs/benchmarks.md sections 101.4 and 102). A stack of
/// layers (RmsNorm, a projection or a routed expert product, the residual add) runs on 9-16 rows (the register-streaming FP8 decode
/// GEMM, the generated skinny GEMMs) and on more than 16 (the decode and prompt tiles of the hand-written GEMMs), and every row must
/// match the same stack run on that row alone, which takes other kernels (the one-row GEMVs). Every layer rewrites the same scratch
/// activations, and the kernels launch as programmatic dependents: a kernel that read its activations through the non-coherent L1
/// path would see the previous layer's rows (stale lines survive <c>griddepcontrol.wait</c>), which is how batched Qwen3-8B-FP8 gave
/// garbage from section 84 to 101 while single sequences were right. Two batched runs must also agree bit for bit.
/// </summary>
public sealed partial class BatchedDecodeRowsTests
{
    private const int Hidden = 1024;

    private const int Layers = 8;

    private const int Experts = 8;

    private const int Routes = 2;

    [Theory]
    [InlineData("fp8", 9)]
    [InlineData("fp8", 12)]
    [InlineData("fp8", 16)]
    [InlineData("fp8", 24)]
    [InlineData("fp8", 40)]
    [InlineData("bf16", 12)]
    [InlineData("bf16", 16)]
    [InlineData("bf16", 40)]
    [InlineData("bf16", 64)]
    [InlineData("mxfp4-moe", 12)]
    [InlineData("mxfp4-moe", 16)]
    [InlineData("mxfp4-moe", 40)]
    [InlineData("fp8-moe", 16)]
    [InlineData("fp8-moe", 40)]
    [InlineData("nvfp4", 9)]
    [InlineData("nvfp4", 16)]
    [InlineData("nvfp4", 40)]
    [InlineData("nvfp4", 64)]
    [InlineData("nvfp4-moe", 12)]
    [InlineData("nvfp4-moe", 40)]
    public void BatchedRowsMatchSingleRows(string format, int rows)
    {
        var provider = new CudaProvider();
        Assert.SkipUnless(provider.CheckAvailability().IsAvailable, "No CUDA device.");
        var info = provider.GetDevices()[0];
        using var device = provider.OpenDevice(info);
        var (graph, x, ids) = BuildStack(format);
        using var model = CompiledModel.Compile(graph, new PlacementOptions
        {
            Devices = [new DeviceSlot { Device = device }],
            Compile = new CompileOptions { MaxTokens = 128, ContextLength = 128 },
            ReservePerDevice = 0,
        });

        var input = new float[rows * Hidden];
        for (int i = 0; i < input.Length; i++)
        {
            input[i] = MathF.Sin(i * 0.37f + i / Hidden) * (1 + i / Hidden % 3);
        }

        var routes = new int[rows * Routes];
        for (int t = 0; t < rows; t++)
        {
            routes[t * Routes] = t % Experts;
            routes[t * Routes + 1] = (t * 3 + 1) % Experts == t % Experts ? (t + 1) % Experts : (t * 3 + 1) % Experts;
        }

        bool fp4 = format.StartsWith("nvfp4", StringComparison.Ordinal);
        var batched = Run(model, x, ids, input, routes, rows);
        Assert.Equal(batched, Run(model, x, ids, input, routes, rows));
        for (int t = 0; t < rows; t++)
        {
            // NVFP4 batches from 9 rows quantize their activations to FP4 per row and a single row runs the 8-bit integer-dot matvec, so
            // their reference is the row repeated as a 9-row batch: every kernel then computes it alone in its row, in the same K order.
            int copies = fp4 ? 9 : 1;
            var single = Run(model, x, ids, Repeat(input.AsSpan(t * Hidden, Hidden), copies), Repeat(routes.AsSpan(t * Routes, Routes), copies), copies);
            double error = 0, norm = 0;
            for (int i = 0; i < Hidden; i++)
            {
                double d = batched[t * Hidden + i] - single[i];
                error += d * d;
                norm += (double)single[i] * single[i];
            }

            // A single row's dense FP8 projections run other kernels than batched rows (the skinny GEMVs and their activation path):
            // after eight layers they part by 2.1 %. Rows that read another layer's activations (section 101.4's stale lines) are
            // 20-50 % away.
            double relative = Math.Sqrt(error / Math.Max(norm, 1e-30));
            double bound = format switch { "fp8" => 5e-2, "nvfp4" or "nvfp4-moe" => 1e-6, _ => 2e-2 };
            Assert.True(relative <= bound, $"{format}, {rows} rows: row {t} is {relative:E2} (relative L2) away from its single-row result");
        }
    }

    /// <summary>
    /// No kernel compiled for dependent launches (compute capability 9.0 and later: every kernel starts with, or issues,
    /// <c>griddepcontrol.wait</c>) reads through the non-coherent path (<c>ld.global.nc</c>, what <c>__ldg</c> and
    /// <c>const __restrict__</c> pointers compile to) except weight streams, which carry <c>L1::no_allocate</c> by convention and are
    /// written by no launch of a step, and the compiler's own constant tables (<c>__cudart_*</c>, the range reduction of sin / cos). A
    /// dependent grid starts while the launches before it still run, so an L1 line it cached through the non-coherent path may hold
    /// rows that an earlier launch rewrites (section 101.4). Checked on the PTX of the decode stacks above at prompt and decode sizes and
    /// of the code tests' workload (an MLP block on k-quants, attention, experts, a convolution).
    /// </summary>
    [Fact]
    public async Task DependentKernelsReadEarlierOutputsCoherently()
    {
        Assert.SkipWhen(NvrtcTranslator.NvrtcVersion is null, "NVRTC is not installed (Sharplight.Cuda.Runtime).");
        var provider = new CudaProvider();
        // A Blackwell workstation GPU's opt-in shared memory, so that the hand-written GEMMs (FP8 prompt and decode forms, f16) serve.
        // With the block-scaled MMAs a CUDA 12.9 driver adds on 12.x, so that the FP4 GEMMs are checked too.
        var profile = CudaCodeTests.Profile(120, 99 * 1024, DeviceFeatures.BlockScaledFp8, DeviceFeatures.BlockScaledFp4);
        var portable = await provider.CompilePortableAsync(new PortableCompilationRequest(profile, (device, artifacts, cancellationToken) =>
        {
            CudaCodeTests.Workload(device, artifacts, cancellationToken);
            foreach (var format in new[] { "fp8", "bf16", "mxfp4-moe", "fp8-moe", "nvfp4", "nvfp4-moe" })
            {
                using var model = CompiledModel.Compile(BuildStack(format).Graph, new PlacementOptions
                {
                    Devices = [new DeviceSlot { Device = device }],
                    Compile = new CompileOptions { MaxTokens = 512, ContextLength = 512, CacheFor = artifacts.For },
                    ReservePerDevice = 0,
                });
            }
        }) { DeviceOptions = new DeviceOptions { UseCompilationCache = false } }, TestContext.Current.CancellationToken);

        Assert.NotEmpty(portable.Entries);
        var offenders = new List<string>();
        foreach (var (key, code) in portable.Entries)
        {
            string ptx = Encoding.UTF8.GetString(code.Span);
            if (!ptx.Contains(CudaPtx.DependentHeader, StringComparison.Ordinal))
            {
                continue;
            }

            var tables = TableRegister().Matches(ptx).Select(m => m.Groups[1].Value).ToHashSet();
            foreach (Match load in NonCoherentLoad().Matches(ptx))
            {
                if (!load.Value.Contains("L1::no_allocate", StringComparison.Ordinal) && !tables.Contains(load.Groups[1].Value))
                {
                    offenders.Add($"{key}: {load.Value.Trim()}");
                }
            }
        }

        Assert.True(offenders.Count == 0, "non-coherent loads in dependent kernels:\n" + string.Join("\n", offenders.Take(20)));
    }

    /// <summary>A register set to the address of one of the compiler's constant tables (<c>mov.u64 %rd1, __cudart_i2opi_f;</c>).</summary>
    [GeneratedRegex(@"mov\.u64\s+(%rd\d+),\s*__cudart_\w+;")]
    private static partial Regex TableRegister();

    /// <summary>A non-coherent global load and its address register.</summary>
    [GeneratedRegex(@"ld\.global\.nc[^\n\[]*\[(%rd\d+)[^\]]*\][^\n]*")]
    private static partial Regex NonCoherentLoad();

    /// <summary>
    /// <see cref="Layers"/> layers on [T, <see cref="Hidden"/>] f32 activations: x += P(RmsNorm(x)) with P a dense projection of the format
    /// or a routed expert product (two experts per row of the "ids" input, equal router weights).
    /// </summary>
    private static (ComputeGraph Graph, Value X, Value? Ids) BuildStack(string format)
    {
        var tg = new TG("batched-decode-rows-" + format);
        var x = tg.InF("x", TG.T, Hidden);
        bool moe = format.EndsWith("-moe", StringComparison.Ordinal);
        var ids = moe ? tg.In("ids", DataType.I32, TG.T, Routes) : null;
        var type = format switch
        {
            "fp8" or "fp8-moe" => DataType.F8E4M3_B128,
            "bf16" => DataType.BF16,
            "nvfp4" or "nvfp4-moe" => DataType.NVFP4_GS,
            _ => DataType.MXFP4,
        };
        var h = x;
        for (int l = 0; l < Layers; l++)
        {
            var n = tg.B.RmsNorm(h, null, 1e-5f);
            Value y;
            if (ids is not null)
            {
                var routed = tg.B.Emit1("MatMulId", [n, tg.RandW($"e{l}", type, 0.05f, Experts, Hidden, Hidden), ids]);
                y = tg.B.Emit1("ReduceSum", [tg.B.Permute(routed, 0, 2, 1)]);
                y = tg.B.Reshape(y, 0, Hidden);
            }
            else
            {
                y = tg.B.MatMul(n, tg.RandW($"w{l}", type, 0.05f, Hidden, Hidden));
            }

            h = tg.B.Add(h, y);
        }

        tg.B.Output("y", h);
        return (tg.B.Build(), x, ids);
    }

    /// <summary><paramref name="copies"/> copies of <paramref name="row"/> back to back.</summary>
    private static T[] Repeat<T>(ReadOnlySpan<T> row, int copies)
    {
        var result = new T[row.Length * copies];
        for (int c = 0; c < copies; c++)
        {
            row.CopyTo(result.AsSpan(c * row.Length));
        }

        return result;
    }

    private static float[] Run(CompiledModel model, Value x, Value? ids, float[] input, int[] routes, int rows)
    {
        model.SetInput<float>(x, input);
        if (ids is not null)
        {
            model.SetInput<int>(ids, routes);
        }

        var parameters = new RunParameters();
        parameters.Symbols["T"] = rows;
        model.Run(parameters);
        var output = new float[rows * Hidden];
        model.GetOutput<float>("y", output);
        return output;
    }
}
