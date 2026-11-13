// BatchBench <model> [--slots 64] [--ctx 512] [--depth 256] [--sizes 1,2,4,8,16,32,64] [--reps 30] [--profile B[,B...]] [--profile-mode logits|greedy|sampled]
//            [--modes logits,greedy,sampled,no-logits] [--profile-top 40]
// <model> is a GGUF, a .slm package or a Hugging Face safetensors directory.
// Prefills every slot with --depth random tokens, then times decode steps of the multi-sequence decoder for each batch size
// (rows = batch size). Modes per step: "logits" = logits downloaded for every row and arg-max taken on the CPU (the path without a
// device sampling head), "greedy" / "sampled" = tokens chosen on the device (temperature 0.8, top-k 40, top-p 0.95 for "sampled";
// only token ids are downloaded), "no-logits" = no output row wanted. Each mode reports wall-clock min / median ms and the
// minimum GPU time of the step (command-buffer timestamps, robust against other load on a shared GPU).
// --profile B prints the per-kernel profile (SHARPLIGHT_VK_PROFILE) of steps with B sequences in --profile-mode (default logits).
// Per batch size it also prints the dispatch / barrier count of the step and the wall time of the first step at that size in the
// first mode (includes recording the command buffers for the new shape) next to the warm minimum.
using System.Diagnostics;
using Sharplight.Gguf;
using Sharplight.Graph;
using Sharplight.HuggingFace;
using Sharplight.Llm;
using Sharplight.Llm.Serving;
using Sharplight.Package;
using Sharplight.Runtime;
using Sharplight.Vulkan;

var argv = args.ToList();
string? Opt(string n, string? d = null) { int i = argv.IndexOf(n); return i >= 0 && i + 1 < argv.Count ? argv[i + 1] : d; }
ProviderRegistry.Register(() => new VulkanProvider());
GgufImporter.Register();
HuggingFaceImporter.Register();
ModelPackageImporter.Register();
int slots = int.Parse(Opt("--slots", "64")!), ctx = int.Parse(Opt("--ctx", "512")!), depth = int.Parse(Opt("--depth", "256")!);
int reps = int.Parse(Opt("--reps", "30")!);
var sizes = Opt("--sizes", "1,2,4,8,16,32,64")!.Split(',').Select(int.Parse).Where(s => s <= slots).ToArray();
var provider = ProviderRegistry.Discover().First(x => x.Availability.IsAvailable && x.Provider.Name != "cpu").Provider;
using var device = provider.OpenDevice(provider.GetDevices()[0], new DeviceOptions());
using var model = LlmModel.Load(argv[0], new LlmLoadOptions { Devices = [new DeviceSlot { Device = device }], ContextLength = ctx, MaxBatchTokens = 512 });
using var dec = (MultiSequenceDecoder)model.CreateBatchDecoder(slots, ctx);
var sampler = (IDeviceSamplingDecoder)dec;
Console.WriteLine($"decoder: {dec.GetType().Name}, slots {dec.SlotCount}, vocab {dec.VocabSize}, device sampling: {sampler.SamplesOnDevice}");
var rng = new Random(1);
int[] Toks(int n) => Enumerable.Range(0, n).Select(_ => rng.Next(1000, 30000)).ToArray();
for (int s = 0; s < slots; s++) dec.Step([new SequenceInput(s, Toks(depth), false)]);   // prefill each slot
double Median(List<double> l) { l.Sort(); return l[l.Count / 2]; }
var segments = dec.Compiled.Segments.OfType<VulkanSegment>().ToList();
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
        gpu = Math.Min(gpu, segments.Sum(s => s.LastGpuMs));
    }
    return (wall.Min(), Median(wall), gpu);
}

string[] modes = Opt("--modes", "logits,greedy,sampled,no-logits")!.Split(',');
var profileSizes = (Opt("--profile") ?? "").Split(',', StringSplitOptions.RemoveEmptyEntries).Select(int.Parse).ToHashSet();
int profileTop = int.Parse(Opt("--profile-top", "40")!);
Console.WriteLine("Per mode: wall ms min / median, GPU ms min; first = wall ms of the first step at this batch size (records the shape); dispatches / barriers per step");
Console.WriteLine($"| batch | first ms | dispatches / barriers | {string.Join(" | ", modes)} |");
Console.WriteLine($"|---|---|---|{string.Concat(modes.Select(_ => "---|"))}");
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
        foreach (var seg in segments) { seg.Profile = true; seg.ResetProfile(); }
        for (int i = 0; i < 20; i++) Step(mode, inputs);
        foreach (var seg in segments) { Console.WriteLine($"--- profile batch {b} ({mode})\n{seg.ProfileReport(profileTop)}"); seg.Profile = false; }
    }
}
