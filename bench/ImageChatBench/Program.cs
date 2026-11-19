// ImageChatBench <model> [--mmproj F] [--streams 16] [--tokens 128] [--slots 16] [--ctx 2048] [--image-tokens 256] [--mode single,engine]
//                [--same-image] [--rounds 2] [--mtp K] [--prompt TEXT] [--image PNG]
// <model> is a GGUF (with --mmproj), a .slm package or a Hugging Face directory of a model with a vision tower.
// Every stream is one image chat (a distinct synthetic 448x448 image per stream unless --same-image, the user turn "<image> --prompt"),
// all submitted at once, each generating --tokens tokens greedily (end-of-generation ignored).
//   single: the path image chats took before the engine had media rows: one session of the multimodal graph with the vision encoder
//           (a Generator), the streams one after another in submission order (TTFT of stream i includes the streams before it).
//   engine: the batching engine over a decoder with media input (--slots sequences, paged KV cache with prefix caching), image prompts
//           built with ImagePrompts (images encoded by the engine's encoder stage, cached by content).
// --rounds N repeats the same streams N times on the same engine (round 2+ hits the prefix cache and the encoder-output cache).
// --mtp K adds the model's multi-token-prediction head with K drafts (engine mode).
// Prints per round: wall time, aggregate generated tok/s, TTFT p50 / p95, mean inter-token latency, images encoded, cached prompt tokens.
using System.Diagnostics;
using Microsoft.Extensions.DependencyInjection;
using Sharplight;
using Sharplight.Imaging;
using Sharplight.Llm;
using Sharplight.Llm.Serving;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Templating;
using Sharplight.Vision;

var argv = args.ToList();
string? Opt(string n, string? d = null)
{
    int i = argv.IndexOf(n);
    return i >= 0 && i + 1 < argv.Count ? argv[i + 1] : d;
}

int streams = int.Parse(Opt("--streams", "16")!);
int tokens = int.Parse(Opt("--tokens", "128")!);
int slots = int.Parse(Opt("--slots", streams.ToString())!);
int ctx = int.Parse(Opt("--ctx", "2048")!);
int imageTokens = int.Parse(Opt("--image-tokens", "256")!);
int rounds = int.Parse(Opt("--rounds", "1")!);
int mtp = int.Parse(Opt("--mtp", "0")!);
bool same = argv.Contains("--same-image");
string question = Opt("--prompt", "Describe this image in detail.")!;
var modes = Opt("--mode", "single,engine")!.Split(',');

using var services = SharplightDefaults.CreateServiceProvider();
var loader = services.GetRequiredService<ModelLoader>();
var provider = services.GetRequiredService<ExecutionProviderCatalog>().Discover().First(x => x.Availability.IsAvailable && x.Provider.Name != "cpu").Provider;
using var device = provider.OpenDevice(provider.GetDevices()[0], new DeviceOptions());
var load = new LlmLoadOptions
{
    Devices = [new DeviceSlot { Device = device }], ContextLength = ctx, MaxBatchTokens = 2048, ImageTokens = new ImageTokenLimits(null, imageTokens),
};
if (Opt("--mmproj") is { } mmproj)
{
    load.Auxiliary[AuxiliaryRoles.Mmproj] = mmproj;
}

using var model = LlmModel.Load(loader, argv[0], load);
using var vision = model.CreateVisionEncoder();
var template = new ChatTemplate(model.ChatTemplate!, "", "");
string prompt = template.Render([new ChatMessage("user", Parts: [ChatContentPart.Image(), ChatContentPart.FromText(question)])], addGenerationPrompt: true,
    new Dictionary<string, object?> { ["enable_thinking"] = false });
var baseImage = Opt("--image") is { } png ? PngCodec.Read(png) : null;
var images = Enumerable.Range(0, streams).Select(i => same ? Image(0) : Image(i)).ToList();
Console.WriteLine($"{Path.GetFileName(argv[0])}: {streams} streams x {tokens} tokens, image {images[0].Width}x{images[0].Height} -> {vision.Preprocess(images[0]).TokenCount} rows, "
    + $"prompt {prompt.Length} chars, device {device.Info.Name}");

foreach (var mode in modes)
{
    if (mode == "single")
    {
        RunSingle();
    }
    else
    {
        await RunEngine();
    }
}

ImageInput Image(int i)
{
    if (baseImage is not null)
    {
        return i == 0 ? baseImage : Shift(baseImage, i);
    }

    const int Side = 448;
    var rgb = new byte[Side * Side * 3];
    var r = new Random(i + 1);
    double fx = 0.01 + r.NextDouble() * 0.05, fy = 0.01 + r.NextDouble() * 0.05, phase = r.NextDouble() * 6;
    for (int y = 0; y < Side; y++)
    {
        for (int x = 0; x < Side; x++)
        {
            int o = (y * Side + x) * 3;
            rgb[o] = (byte)(127 + 127 * Math.Sin(fx * x + phase));
            rgb[o + 1] = (byte)(127 + 127 * Math.Sin(fy * y + 2 * phase));
            rgb[o + 2] = (byte)((x / 64 + y / 64 + i) % 2 == 0 ? 220 : 40);
        }
    }

    return new ImageInput(Side, Side, rgb);
}

static ImageInput Shift(ImageInput image, int by)
{
    var rgb = new byte[image.Rgb.Length];
    for (int k = 0; k < rgb.Length; k++)
    {
        rgb[k] = (byte)(image.Rgb[k] ^ (by * 37));
    }

    return new ImageInput(image.Width, image.Height, rgb);
}

void RunSingle()
{
    using var session = model.CreateSession(WellKnown.DecodeMultimodalGraph);
    var generator = new Generator(session, model.Tokenizer, vision);
    var options = new GenerationOptions { MaxNewTokens = tokens, Sampling = SamplingOptions.Greedy, StopAtEog = false };
    var ttft = new List<double>();
    var itl = new List<double>();
    int generated = 0;
    var wall = Stopwatch.StartNew();
    foreach (var image in images)
    {
        session.Reset();
        double last = 0;
        bool first = true;
        foreach (var t in generator.Generate(prompt, [image], options))
        {
            double now = wall.Elapsed.TotalMilliseconds;
            if (t.Id < 0)
            {
                continue;
            }

            if (first)
            {
                ttft.Add(now);
                first = false;
            }
            else
            {
                itl.Add(now - last);
            }

            last = now;
            generated++;
        }
    }

    Report("single session", wall.Elapsed, generated, ttft, itl, 0, 0);
}

async Task RunEngine()
{
    var speculator = mtp > 0 && model.HasDraftHead ? model.CreateMtpSpeculator(slots, mtp, ctx, mediaInput: true) : null;
    int? logitsRows = speculator is null ? null : slots + Math.Min(slots, 16) * mtp;
    var decoder = model.CreateBatchDecoder(slots, ctx, maxLogitsRows: logitsRows, draftTokens: speculator?.MaxDraftTokens ?? 0, mediaInput: true);
    using var engine = new InferenceEngine(decoder, model.Tokenizer, new EngineOptions { Speculator = speculator });
    using var builder = new ImagePrompts(vision, model.Tokenizer);
    var promptTokens = model.Tokenizer.Encode(prompt, addSpecial: true, parseSpecial: true);
    for (int round = 1; round <= rounds; round++)
    {
        var before = engine.GetMetrics();
        var ttft = new List<double>();
        var itl = new List<double>();
        int generated = 0;
        int cached = 0;
        var wall = Stopwatch.StartNew();
        // Building a request preprocesses and hashes its image on the caller's thread, as the chat service does.
        var tasks = images.Select(image => Task.Run(async () =>
        {
            var request = builder.WithImages(new GenerationRequest
            {
                PromptTokens = promptTokens, MaxTokens = tokens, StopAtEog = false, Sampling = SamplingParams.GreedyDefault,
            }, [image]);
            var handle = engine.Submit(request);
            double last = 0;
            bool first = true;
            var gaps = new List<double>();
            double firstMs = 0;
            GenerationUsage usage = default;
            await foreach (var chunk in handle.ReadAllAsync())
            {
                double now = wall.Elapsed.TotalMilliseconds;
                if (chunk.Usage is { } u)
                {
                    usage = u;
                }

                if (chunk.Text.Length == 0)
                {
                    continue;
                }

                if (first)
                {
                    firstMs = now;
                    first = false;
                }
                else
                {
                    gaps.Add(now - last);
                }

                last = now;
            }

            return (firstMs, gaps, usage);
        })).ToList();
        foreach (var (firstMs, gaps, usage) in await Task.WhenAll(tasks))
        {
            ttft.Add(firstMs);
            itl.AddRange(gaps);
            generated += usage.CompletionTokens;
            cached += usage.CachedTokens;
        }

        var after = engine.GetMetrics();
        Report($"engine ({slots} slots{(speculator is null ? "" : $", MTP {mtp}")}), round {round}", wall.Elapsed, generated, ttft, itl,
            after.MediaEncoded - before.MediaEncoded, cached, after.DraftAcceptanceRate);
    }
}

static void Report(string name, TimeSpan wall, int generated, List<double> ttft, List<double> itl, long encoded, int cached, double acceptance = 0)
{
    ttft.Sort();
    double P(List<double> v, double q) => v.Count == 0 ? 0 : v[Math.Min(v.Count - 1, (int)(q * v.Count))];
    Console.WriteLine($"{name}: wall {wall.TotalSeconds:F2} s, {generated / wall.TotalSeconds:F1} tok/s aggregate, TTFT p50 {P(ttft, 0.5):F0} ms p95 {P(ttft, 0.95):F0} ms, "
        + $"ITL mean {(itl.Count == 0 ? 0 : itl.Average()):F1} ms, images encoded {encoded}, cached prompt tokens {cached}"
        + (acceptance > 0 ? $", draft acceptance {acceptance:P0}" : ""));
}
