// BatchBench <model> [--slots 64] [--ctx 512] [--depth 256] [--sizes 1,2,4,8,16,32,64] [--reps 30] [--profile B[,B...]] [--profile-mode logits|greedy|sampled]
//            [--modes logits,greedy,sampled,no-logits] [--profile-top 40] [--batch 512] [--mixed D+P[x N],...] [--text] [--feedback]
// <model> is a GGUF, a .slm package or a Hugging Face safetensors directory.
// Prefills every slot with --depth random tokens, then times decode steps of the multi-sequence decoder for each batch size
// (rows = batch size). Modes per step: "logits" = logits downloaded for every row and arg-max taken on the CPU (the path without a
// device sampling head), "greedy" / "sampled" = tokens chosen on the device (temperature 0.8, top-k 40, top-p 0.95 for "sampled";
// only token ids are downloaded), "no-logits" = no output row wanted. Each mode reports wall-clock min / median ms and the
// minimum GPU time of the step (command-buffer timestamps, robust against other load on a shared GPU).
// --profile B prints the per-kernel profile (SHARPLIGHT_VK_PROFILE) of steps with B sequences in --profile-mode (default logits).
// Per batch size it also prints the dispatch / barrier count of the step and the wall time of the first step at that size in the
// first mode (includes recording the command buffers for the new shape) next to the warm minimum.
// --mixed D+P,... times steps that mix D decode rows (slots 0..D-1) with prompt chunks of P rows (a fresh sequence from position 0
// in each of the last slots, as the serving engine's chunked prefill feeds them); "D+PxN" splits P*N prompt rows over N sequences.
// --batch sets the decoder's rows per step (default 512). With --mixed, --profile takes the index of the mixed entry instead.
// --text: real-text routing instead of random tokens. Each slot is prefilled with a tokenized prompt of random English words (the
// vocabulary of bench/loadgen.py, the serving benchmark's prompts) and, in the greedy / sampled modes, every decode step feeds back the
// tokens the previous step chose for its rows (the serving engine's decode steps), so mixture-of-experts layers route as they do on text.
// --feedback: random-token prompts, but the chosen tokens fed back as with --text (the workload of bench/s54_vllm_decode_profile.py).
// --provider vulkan|cuda picks the GPU provider (default: the first available one, Vulkan); profiles of CUDA steps need
// SHARPLIGHT_CUDA_PROFILE-style per-launch timing, which --profile switches on per segment like on Vulkan.
using System.Diagnostics;
using Sharplight;
using Microsoft.Extensions.DependencyInjection;
using Sharplight.Graph;
using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Llm.Serving;
using Sharplight.Package;
using Sharplight.Runtime;
using Sharplight.Cuda;
using Sharplight.Vulkan;

var argv = args.ToList();
string? Opt(string n, string? d = null)
{
    int i = argv.IndexOf(n);
    return i >= 0 && i + 1 < argv.Count ? argv[i + 1] : d;
}

using var services = SharplightDefaults.CreateServiceProvider(s => s.AddCudaProvider());
var loader = services.GetRequiredService<ModelLoader>();
int slots = int.Parse(Opt("--slots", "64")!);
int ctx = int.Parse(Opt("--ctx", "512")!);
int depth = int.Parse(Opt("--depth", "256")!);
int reps = int.Parse(Opt("--reps", "30")!);
var sizes = Opt("--sizes", "1,2,4,8,16,32,64")!.Split(',').Select(int.Parse).Where(s => s <= slots).ToArray();
string? providerName = Opt("--provider");
var provider = services.GetRequiredService<ExecutionProviderCatalog>().Discover()
    .First(x => x.Availability.IsAvailable && x.Provider.Name != "cpu" && (providerName is null || x.Provider.Name == providerName)).Provider;
using var device = provider.OpenDevice(provider.GetDevices()[0], new DeviceOptions());
// --kv-type f16|q8_0|...: element type of the KV caches (default: the model's).
Sharplight.Tensors.DataType? kvType = Opt("--kv-type") is { } kt ? Sharplight.Tensors.DataType.Get(kt) : null;
using var model = LlmModel.Load(loader, argv[0], new LlmLoadOptions
{
    Devices = [new DeviceSlot { Device = device }], ContextLength = ctx, MaxBatchTokens = int.Parse(Opt("--batch", "512")!), KvCacheType = kvType,
});
// --kv-tokens N: paged pool size (0: the contiguous per-slot cache; default: the paged default of slots x ctx).
long? kvTokens = Opt("--kv-tokens") is { } kv ? long.Parse(kv) : null;
using var dec = (MultiSequenceDecoder)model.CreateBatchDecoder(slots, ctx, kvCacheTokens: kvTokens);
var sampler = (IDeviceSamplingDecoder)dec;
Console.WriteLine($"decoder: {dec.GetType().Name}, slots {dec.SlotCount}, vocab {dec.VocabSize}, device sampling: {sampler.SamplesOnDevice}, "
    + $"KV {(dec.KvBlockSize > 0 ? $"paged {dec.KvBlocksTotal} x {dec.KvBlockSize}" : "contiguous")}");
var rng = new Random(1);
bool text = argv.Contains("--text");
bool feedback = text || argv.Contains("--feedback");
string[] words = ("time year people way day man thing woman life child world school state family student group country problem hand part "
    + "place case week company system program question work government number night point home water room mother area money story "
    + "fact month lot right study book eye job word business issue side kind head house service friend father power hour game line "
    + "end member law car city community name president team minute idea kid body information back parent face others level office "
    + "door health person art war history party result change morning reason research girl guy moment air teacher force education").Split(' ');
int[] Words(int n)
{
    var ids = new List<int>();
    while (ids.Count < n)
    {
        ids.AddRange(model.Tokenizer.Encode(string.Join(' ', Enumerable.Range(0, 64).Select(_ => words[rng.Next(words.Length)])) + " ", addSpecial: false));
    }

    return [.. ids.Take(n)];
}
int[] Toks(int n) => text ? Words(n) : Enumerable.Range(0, n).Select(_ => rng.Next(1000, 30000)).ToArray();
// Prefill each slot a timed batch size reaches, in chunks of the decoder's rows per step (long depths). Slots above the largest size
// stay empty, so one sequence at a long depth fits a paged pool (--kv-tokens) sized for it.
int chunk = int.Parse(Opt("--batch", "512")!);
int usedSlots = Opt("--mixed") is null ? sizes.Max() : slots;
for (int s = 0; s < usedSlots; s++)
{
    for (int done = 0; done < depth; done += chunk)
    {
        dec.Step([new SequenceInput(s, Toks(Math.Min(chunk, depth - done)), false)]);
    }
}
double Median(List<double> l)
{
    l.Sort();
    return l[l.Count / 2];
}

var segments = dec.Compiled.Segments.OfType<VulkanSegment>()
    .Select(v => new SegmentView(() => v.LastGpuMs, on => v.Profile = on, v.ResetProfile, v.ProfileReport, v.Stats))
    .Concat(dec.Compiled.Segments.OfType<CudaSegment>()
        .Select(c => new SegmentView(() => c.LastGpuMs, on => c.Profile = on, c.ResetProfile, c.ProfileReport, c.Stats)))
    .ToList();
var tokens = new int[slots];
var logprobs = new float[slots];
uint counter = 0;

void Step(string mode, List<SequenceInput> inputs)
{
    switch (mode)
    {
        case "logits":
            foreach (var l in dec.Step(inputs)) Sampler.ArgMax(l.Span);
            break;
        case "no-logits":
            dec.Step([.. inputs.Select(i => i with { WantLogits = false })]);
            break;
        default:
            var row = mode == "greedy" ? SamplingRow.Greedy() : new SamplingRow { Temperature = 0.8f, TopK = 40, TopP = 0.95f, Seed = 7, Counter = counter++ };
            sampler.Step(inputs, [.. inputs.Select(_ => (SamplingRow?)row)], tokens, logprobs);
            if (feedback)
            {
                // Feed the chosen tokens back as the rows' next inputs (single-token decode rows only).
                for (int i = 0; i < inputs.Count; i++)
                {
                    if (inputs[i].Tokens.Length == 1)
                    {
                        inputs[i] = inputs[i] with { Tokens = new[] { tokens[i] } };
                    }
                }
            }
            break;
    }
}

(double Min, double Median, double GpuMin) Time(string mode, List<SequenceInput> inputs)
{
    for (int i = 0; i < 5; i++) Step(mode, inputs);
    var wall = new List<double>();
    double gpu = double.MaxValue;
    for (int i = 0; i < reps; i++)
    {
        var sw = Stopwatch.StartNew();
        Step(mode, inputs);
        wall.Add(sw.Elapsed.TotalMilliseconds);
        gpu = Math.Min(gpu, segments.Sum(s => s.Gpu()));
    }
    return (wall.Min(), Median(wall), gpu);
}

string[] modes = Opt("--modes", "logits,greedy,sampled,no-logits")!.Split(',');
var profileSizes = (Opt("--profile") ?? "").Split(',', StringSplitOptions.RemoveEmptyEntries).Select(int.Parse).ToHashSet();
int profileTop = int.Parse(Opt("--profile-top", "40")!);
Console.WriteLine("Per mode: wall ms min / median, GPU ms min; first = wall ms of the first step at this batch size (records the shape); dispatches / barriers per step");
Console.WriteLine($"| batch | first ms | dispatches / barriers | {string.Join(" | ", modes)} |");
Console.WriteLine($"|---|---|---|{string.Concat(modes.Select(_ => "---|"))}");
string? mixed = Opt("--mixed");
if (mixed is not null)
{
    string mode = modes[0];
    Console.WriteLine($"| decode rows | prompt rows (sequences) | first ms | dispatches / barriers | {mode} wall min / median, GPU min | GPU ms per prompt row over decode-only |");
    Console.WriteLine("|---|---|---|---|---|---|");
    var baseline = new Dictionary<int, double>();
    int entry = 0;
    foreach (string spec in mixed.Split(','))
    {
        var parts = spec.Split('+');
        int d = int.Parse(parts[0]);
        var pp = parts[1].Split('x');
        int p = int.Parse(pp[0]);
        int n = pp.Length > 1 ? int.Parse(pp[1]) : (p > 0 ? 1 : 0);
        List<SequenceInput> Inputs()
        {
            var list = Enumerable.Range(0, d).Select(s => new SequenceInput(s, new[] { rng.Next(1000, 30000) }, true)).ToList();
            for (int j = 0; j < n; j++)
            {
                int slot = slots - 1 - j;
                dec.Reset(slot);
                list.Add(new SequenceInput(slot, Toks(p), true));
            }

            return list;
        }

        // Fresh prompt sequences each step: Reset is free and the next input starts at position 0.
        var first = Stopwatch.StartNew();
        Step(mode, Inputs());
        double firstMs = first.Elapsed.TotalMilliseconds;
        var stats = segments.Select(s => s.Stats(d + p * n)).Aggregate((0, 0), (a, x) => (a.Item1 + x.Dispatches, a.Item2 + x.Barriers));
        for (int i = 0; i < 3; i++) Step(mode, Inputs());
        var wall = new List<double>();
        double gpu = double.MaxValue;
        for (int i = 0; i < reps; i++)
        {
            var inputs = Inputs();
            var sw = Stopwatch.StartNew();
            Step(mode, inputs);
            wall.Add(sw.Elapsed.TotalMilliseconds);
            gpu = Math.Min(gpu, segments.Sum(s => s.Gpu()));
        }

        if (n == 0) baseline[d] = gpu;
        string perRow = n > 0 && baseline.TryGetValue(d, out double b0) ? $"{(gpu - b0) / (p * n):F3}" : "";
        Console.WriteLine($"| {d} | {p * n} ({n}) | {firstMs:F1} | {stats.Item1} / {stats.Item2} | {wall.Min():F2} / {Median(wall):F2}, {gpu:F2} | {perRow} |");
        if (profileSizes.Contains(entry))
        {
            foreach (var seg in segments)
            {
                seg.SetProfile(true);
                seg.Reset();
            }

            for (int i = 0; i < 10; i++) Step(mode, Inputs());
            foreach (var seg in segments)
            {
                Console.WriteLine($"--- profile mixed {spec} ({mode})\n{seg.Report(profileTop)}");
                seg.SetProfile(false);
            }
        }

        entry++;
    }

    return;
}

foreach (int b in sizes)
{
    var inputs = Enumerable.Range(0, b).Select(s => new SequenceInput(s, new[] { rng.Next(1000, 30000) }, true)).ToList();
    var first = Stopwatch.StartNew();
    Step(modes[0], inputs);
    double firstMs = first.Elapsed.TotalMilliseconds;
    var stats = segments.Select(s => s.Stats(b)).Aggregate((0, 0), (a, x) => (a.Item1 + x.Dispatches, a.Item2 + x.Barriers));
    var cells = modes.Select(m => Time(m, inputs)).Select(t => $"{t.Min:F2} / {t.Median:F2}, {t.GpuMin:F2}").ToList();
    Console.WriteLine($"| {b} | {firstMs:F1} | {stats.Item1} / {stats.Item2} | {string.Join(" | ", cells)} |");
    if (profileSizes.Contains(b))
    {
        string mode = Opt("--profile-mode", "logits")!;
        foreach (var seg in segments)
        {
            seg.SetProfile(true);
            seg.Reset();
        }

        for (int i = 0; i < 20; i++) Step(mode, inputs);
        foreach (var seg in segments)
        {
            Console.WriteLine($"--- profile batch {b} ({mode})\n{seg.Report(profileTop)}");
            seg.SetProfile(false);
        }
    }
}
