// VideoEnhancementBench --model <seedvr2 folder or .slm> --input <clip.mp4 | frames.u8> [--input-shape FxHxW] --width W --height H
//                       [--provider cuda|vulkan|cpu] [--batch N] [--runs 5] [--warmup 1] [--reduced-precision]
//                       [--output-u8 out.u8] [--output-video out.mp4] [--json results.json] [--noise <reference dump folder>]
// Times IVideoEnhancer runs end to end: decode the input file, enhance (per phase: encoding = resize + VAE encoder, restoring = DiT,
// decoding = VAE decoder + colour correction), encode the output file; after the warm-up runs (kernel compilation, weight upload) the
// median of the timed runs is reported. The frames of the last run can be written raw for a PSNR check against another implementation;
// --noise takes the initial noise and the encoder sample of a tools/seedvr2-reference.py dump, so the run reproduces that one.
// --profile FxHxW (padded frames x height x width) times every kernel launch of each graph instead (Vulkan and CUDA).
using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Sharplight;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;
using Sharplight.Video;
using Sharplight.VideoEnhancement;
using Sharplight.Vision;

var options = Options.Parse(args);
using var services = SharplightDefaults.CreateServiceProvider(s => s.AddCudaProvider());
var providers = services.GetRequiredService<ExecutionProviderCatalog>();
if (!providers.TryFind(options.Provider, out var provider))
{
    Console.Error.WriteLine($"provider '{options.Provider}' is not available");
    return 2;
}

using var device = provider.OpenDevice(provider.GetDevices()[0]);
var load = Stopwatch.StartNew();
using var model = TaskModel.Load(services.GetRequiredService<ModelLoader>(), new TaskLoadOptions
{
    ModelPath = options.Model,
    Devices = [new DeviceSlot { Device = device }],
    AllowReducedPrecision = options.ReducedPrecision,
    Log = message => Console.Error.WriteLine($"{DateTime.Now:HH:mm:ss.fff} {message.Split('\n')[0]}"),
});
double loadSeconds = load.Elapsed.TotalSeconds;
if (options.Profile is { } profile)
{
    int[] size = [.. profile.Split('x').Select(int.Parse)];
    GraphProfiler.Run(model, size[0], size[1], size[2], 30);
    return 0;
}

using var enhancer = services.GetRequiredService<TaskClientProvider>().Create<IVideoEnhancer>(model);
var encoders = services.GetRequiredService<VideoEncoderCatalog>();
var decoders = services.GetRequiredService<VideoDecoderCatalog>();
Console.WriteLine($"{model.Definition.Name} on {provider.Name} ({device.Info.Name}), loaded in {loadSeconds:F1} s");

var runs = new List<Dictionary<string, double>>();
VideoEnhancementResult? last = null;
for (int run = 0; run < options.Warmup + options.Runs; run++)
{
    var times = new Dictionary<string, double>();
    var total = Stopwatch.StartNew();
    var read = Stopwatch.StartNew();
    var (frames, fps) = await ReadAsync(decoders, options);
    times["read"] = read.Elapsed.TotalSeconds;
    var phases = new Dictionary<VideoEnhancementPhase, double>();
    var enhance = Stopwatch.StartNew();
    await foreach (var update in enhancer.EnhanceStreamingAsync(new VideoEnhancementRequest { Frames = frames, FramesPerSecond = fps },
        new VideoEnhancementOptions
        {
            Width = options.Width,
            Height = options.Height,
            FramesPerBatch = options.Batch,
            Seed = 42,
            InitialNoise = options.Noise is { } noise ? ReadFloats(Path.Combine(noise, "noise.f32")) : null,
            EncoderNoise = options.Noise is { } encoder ? ReadFloats(Path.Combine(encoder, "latent_eps.f32")) : null,
        }))
    {
        phases.TryAdd(update.Phase, enhance.Elapsed.TotalSeconds);
        last = update.Result ?? last;
    }

    times["encoding"] = phases[VideoEnhancementPhase.Restoring] - phases[VideoEnhancementPhase.Encoding];
    times["restoring"] = phases[VideoEnhancementPhase.Decoding] - phases[VideoEnhancementPhase.Restoring];
    times["decoding"] = phases[VideoEnhancementPhase.Completed] - phases[VideoEnhancementPhase.Decoding];
    times["enhance"] = enhance.Elapsed.TotalSeconds;
    var write = Stopwatch.StartNew();
    if (options.OutputVideo is { } video)
    {
        await encoders.SaveAsync(last!.ToClip(), video);
    }

    times["write"] = write.Elapsed.TotalSeconds;
    times["total"] = total.Elapsed.TotalSeconds;
    string label = run < options.Warmup ? "warm-up" : "timed";
    Console.WriteLine($"{label} {string.Join(", ", times.Select(t => string.Create(CultureInfo.InvariantCulture, $"{t.Key} {t.Value:F2} s")))}");
    if (run >= options.Warmup)
    {
        runs.Add(times);
    }
}

var median = runs[0].Keys.ToDictionary(k => k, k => Median(runs.Select(r => r[k])));
Console.WriteLine($"median of {runs.Count}: {string.Join(", ", median.Select(t => string.Create(CultureInfo.InvariantCulture, $"{t.Key} {t.Value:F2} s")))}");
if (options.OutputU8 is { } raw)
{
    using var stream = File.Create(raw);
    foreach (var frame in last!.Frames)
    {
        stream.Write(frame.Rgb);
    }
}

if (options.Json is { } json)
{
    File.WriteAllText(json, JsonSerializer.Serialize(new Dictionary<string, object>
    {
        ["model"] = model.Definition.Name,
        ["provider"] = provider.Name,
        ["device"] = device.Info.Name,
        ["load_seconds"] = loadSeconds,
        ["frames"] = last!.Frames.Count,
        ["output"] = $"{last.Width}x{last.Height}",
        ["runs"] = runs,
        ["median"] = median,
    }, new JsonSerializerOptions { WriteIndented = true }));
}

return 0;

static double Median(IEnumerable<double> values)
{
    var sorted = values.Order().ToArray();
    return sorted.Length % 2 == 1 ? sorted[sorted.Length / 2] : (sorted[sorted.Length / 2 - 1] + sorted[sorted.Length / 2]) / 2;
}

static float[] ReadFloats(string path)
{
    var bytes = File.ReadAllBytes(path);
    var values = new float[bytes.Length / sizeof(float)];
    Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
    return values;
}

static async Task<(List<ImageInput> Frames, double Fps)> ReadAsync(VideoDecoderCatalog decoders, Options options)
{
    if (options.Input.EndsWith(".u8", StringComparison.OrdinalIgnoreCase))
    {
        var bytes = await File.ReadAllBytesAsync(options.Input);
        int[] shape = [.. options.InputShape!.Split('x').Select(int.Parse)];
        int frameBytes = shape[1] * shape[2] * 3;
        return ([.. Enumerable.Range(0, shape[0]).Select(f => new ImageInput(shape[2], shape[1], bytes.AsSpan(f * frameBytes, frameBytes).ToArray()))], 24);
    }

    var clip = await decoders.ReadClipAsync(options.Input);
    return ([.. clip.Frames], clip.FramesPerSecond);
}
