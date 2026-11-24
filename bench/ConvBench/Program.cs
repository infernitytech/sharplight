// ConvBench [--provider cuda|vulkan] [--strict] [--x-type f32|f16] [--weight-type f32|f16] [--iters 20] [--filter text] [--cases "label:N:C:H:W:O:k:s[:parts]" ...]
// GPU time of single 2-D convolutions (bias, "same" padding) compiled alone: the sum over the graph's launches of each launch's
// minimum time across the timed runs (the GPU is shared, so minimums), with the TFLOPS of the convolution's multiply-adds. A case with
// parts > 1 convolves the channel concatenation of that many inputs (the stacked frame windows of a causal 3-D convolution), so the
// staging of the concatenation is part of its time. Reduced precision unless --strict.
using System.Globalization;
using System.Runtime.InteropServices;
using Microsoft.Extensions.DependencyInjection;
using Sharplight;
using Sharplight.Cuda;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tensors;
using Sharplight.Vulkan;

string provider = "cuda";
bool strict = false;
var xType = DataType.F32;
var weightType = DataType.F32;
int iters = 20;
string? filter = null;
var cases = new List<string>();
int top = 0;
for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--provider": provider = args[++i]; break;
        case "--strict": strict = true; break;
        case "--x-type": xType = DataType.Get(args[++i]); break;
        case "--weight-type": weightType = DataType.Get(args[++i]); break;
        case "--iters": iters = int.Parse(args[++i], CultureInfo.InvariantCulture); break;
        case "--filter": filter = args[++i]; break;
        case "--top": top = int.Parse(args[++i], CultureInfo.InvariantCulture); break;
        case "--cases":
            while (i + 1 < args.Length && !args[i + 1].StartsWith("--", StringComparison.Ordinal))
            {
                cases.Add(args[++i]);
            }

            break;
        default:
            Console.Error.WriteLine($"unknown option {args[i]}");
            return 2;
    }
}

if (cases.Count == 0)
{
    cases.AddRange(DefaultCases.All);
}

using var services = SharplightDefaults.CreateServiceProvider(s => s.AddCudaProvider());
var catalog = services.GetRequiredService<ExecutionProviderCatalog>();
if (!catalog.TryFind(provider, out var executionProvider))
{
    Console.Error.WriteLine($"provider '{provider}' is not available");
    return 2;
}

using var device = executionProvider.OpenDevice(executionProvider.GetDevices()[0]);
Console.WriteLine($"# {executionProvider.Name} {device.Info.Name}, {(strict ? "strict" : "reduced")} precision, x {xType}, weights {weightType}, {iters} runs");
foreach (var spec in cases)
{
    var c = ConvCase.Parse(spec);
    if (filter is not null && !c.Label.Contains(filter, StringComparison.OrdinalIgnoreCase))
    {
        continue;
    }

    var (ms, report) = Measure(device, c, xType, weightType, strict, iters);
    double gflop = c.GigaFlops;
    Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
        $"BENCH {c.Label,-34} {c.N}x{c.C}x{c.H}x{c.W} -> {c.O} k{c.K} s{c.Stride} parts {c.Parts}  {ms,9:F3} ms  {gflop / ms,7:F1} TFLOPS"));
    if (top > 0)
    {
        Console.WriteLine(report);
    }
}

return 0;

static (double Ms, string Report) Measure(IComputeDevice device, ConvCase c, DataType xType, DataType weightType, bool strict, int iters)
{
    var weights = new Dictionary<string, WeightInfo>();
    var b = new GraphBuilder("conv", name => weights.GetValueOrDefault(name));
    var random = new Random(7);
    Value Weight(string name, DataType type, float scale, params long[] dims)
    {
        long count = dims.Aggregate(1L, (a, d) => a * d);
        var values = new float[count];
        for (long i = 0; i < count; i++)
        {
            values[i] = (float)(random.NextDouble() * 2 - 1) * scale;
        }

        byte[] bytes = type == DataType.F16
            ? MemoryMarshal.AsBytes(values.Select(v => (Half)v).ToArray().AsSpan()).ToArray()
            : MemoryMarshal.AsBytes(values.AsSpan()).ToArray();
        weights[name] = new WeightInfo { Name = name, Type = type, Shape = new Shape([.. dims.Select(d => (Dim)d)]), Source = new MemoryWeightSource(bytes) };
        return b.Weight(name);
    }

    var inputs = new List<Value>();
    for (int p = 0; p < c.Parts; p++)
    {
        inputs.Add(b.Input($"x{p}", InputRole.Other, xType, new Shape([c.N, c.C / c.Parts, c.H, c.W])));
    }

    var x = inputs.Count == 1 ? inputs[0] : b.Concat(1, [.. inputs]);
    var w = Weight("w", weightType, 1f / MathF.Sqrt(c.C * c.K * c.K), c.O, c.C, c.K, c.K);
    var bias = Weight("b", DataType.F32, 0.1f, c.O);
    long pad = c.K / 2;
    var y = b.Emit1("Conv", [x, w, bias], Attributes.From(("strides", new long[] { c.Stride, c.Stride }), ("pads", new[] { pad, pad, pad, pad }),
        ("dilations", new long[] { 1, 1 }), ("groups", 1L)));
    b.Output("y", y);
    using var model = CompiledModel.Compile(b.Build(), new PlacementOptions
    {
        Devices = [new DeviceSlot { Device = device }],
        Compile = new CompileOptions { MaxTokens = 1, AllowReducedPrecision = !strict },
        ReservePerDevice = 0,
    });
    foreach (var input in inputs)
    {
        long count = c.N * (c.C / c.Parts) * c.H * c.W;
        var values = new float[count];
        for (long i = 0; i < count; i++)
        {
            values[i] = (float)(random.NextDouble() * 2 - 1);
        }

        if (xType == DataType.F16)
        {
            model.SetInput<Half>(input, values.Select(v => (Half)v).ToArray());
        }
        else
        {
            model.SetInput<float>(input, values);
        }
    }

    var parameters = new RunParameters();
    model.Run(parameters);
    foreach (var segment in model.Segments)
    {
        SetProfile(segment, true);
    }

    for (int i = 0; i < iters; i++)
    {
        model.Run(parameters);
    }

    double ms = 0;
    var report = new System.Text.StringBuilder();
    foreach (var segment in model.Segments)
    {
        var (data, text) = Profile(segment);
        ms += data.Values.Sum(v => v.MinMs * v.Count / iters);
        report.Append(text);
        SetProfile(segment, false);
    }

    return (ms, report.ToString());
}

static void SetProfile(ICompiledSegment segment, bool on)
{
    switch (segment)
    {
        case CudaSegment cuda:
            cuda.ResetProfile();
            cuda.Profile = on;
            break;
        case VulkanSegment vulkan:
            vulkan.ResetProfile();
            vulkan.Profile = on;
            break;
    }
}

static (Dictionary<string, (double Ms, int Count, double MinMs)> Data, string Report) Profile(ICompiledSegment segment) => segment switch
{
    CudaSegment cuda => (cuda.ProfileData, cuda.ProfileReport(8)),
    VulkanSegment vulkan => (vulkan.ProfileData, vulkan.ProfileReport(8)),
    _ => ([], ""),
};

/// <summary>One convolution: batch N, C input channels (over Parts concatenated inputs), H x W map, O outputs, K x K kernel.</summary>
internal sealed record ConvCase(string Label, long N, long C, long H, long W, long O, long K, long Stride, int Parts)
{
    public double GigaFlops
    {
        get
        {
            long ho = (H + 2 * (K / 2) - K) / Stride + 1;
            long wo = (W + 2 * (K / 2) - K) / Stride + 1;
            return 2.0 * N * O * ho * wo * C * K * K * 1e-9;
        }
    }

    public static ConvCase Parse(string spec)
    {
        var p = spec.Split(':');
        long L(int i) => long.Parse(p[i], CultureInfo.InvariantCulture);
        return new ConvCase(p[0], L(1), L(2), L(3), L(4), L(5), L(6), L(7), p.Length > 8 ? int.Parse(p[8], CultureInfo.InvariantCulture) : 1);
    }
}

/// <summary>The default shapes: SeedVR2's causal 3-D VAE (3 stacked frame windows, 480 x 832 output), an SDXL VAE decoder at 1024 x 1024, ESRGAN and ResNet layers.</summary>
internal static class DefaultCases
{
    public static readonly string[] All =
    [
        "seedvr2 384>128 480x832:1:384:480:832:128:3:1:3",
        "seedvr2 768>128 480x832:1:768:480:832:128:3:1:3",
        "seedvr2 768>256 240x416:1:768:240:416:256:3:1:3",
        "seedvr2 1536>256 240x416:1:1536:240:416:256:3:1:3",
        "seedvr2 1536>512 120x208:1:1536:120:208:512:3:1:3",
        "seedvr2 1536>512 60x104 x4:4:1536:60:104:512:3:1:3",
        "sdxl-vae 512>512 128x128:1:512:128:128:512:3:1",
        "sdxl-vae 512>512 256x256:1:512:256:256:512:3:1",
        "sdxl-vae 512>512 512x512:1:512:512:512:512:3:1",
        "sdxl-vae 512>256 512x512:1:512:512:512:256:3:1",
        "sdxl-vae 256>256 512x512:1:256:512:512:256:3:1",
        "sdxl-vae 256>128 1024x1024:1:256:1024:1024:128:3:1",
        "sdxl-vae 128>128 1024x1024:1:128:1024:1024:128:3:1",
        "esrgan 64>64 480x832:1:64:480:832:64:3:1",
        "esrgan 192>64 480x832:1:192:480:832:64:3:1",
        "resnet 256>256 14x14 b32:32:256:14:14:256:3:1",
        "resnet 64>64 56x56 b8:8:64:56:56:64:3:1",
        "1x1 512>512 256x256:1:512:256:256:512:1:1",
    ];
}
