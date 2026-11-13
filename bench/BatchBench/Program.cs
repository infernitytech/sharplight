// BatchBench <model> [--slots 64] [--ctx 512] [--depth 256] [--sizes 1,2,4,8,16,32,64] [--profile B]
// Prefills every slot with --depth random tokens, then times decode steps of the multi-sequence decoder for each batch size
// (rows = batch size, logits wanted for all), plus the same step without logits. --profile B prints the per-kernel
// profile (SHARPLIGHT_VK_PROFILE) of decode steps with B sequences, read through reflection (the compiled model is internal).
using System.Diagnostics;
using System.Reflection;
using Sharplight.Gguf;
using Sharplight.Llm;
using Sharplight.Llm.Serving;
using Sharplight.Package;
using Sharplight.Runtime;
using Sharplight.Vulkan;

var argv = args.ToList();
string? Opt(string n, string? d = null) { int i = argv.IndexOf(n); return i >= 0 && i + 1 < argv.Count ? argv[i + 1] : d; }
ProviderRegistry.Register(() => new VulkanProvider());
GgufImporter.Register();
ModelPackageImporter.Register();
int slots = int.Parse(Opt("--slots", "64")!), ctx = int.Parse(Opt("--ctx", "512")!), depth = int.Parse(Opt("--depth", "256")!);
var sizes = Opt("--sizes", "1,2,4,8,16,32,64")!.Split(',').Select(int.Parse).Where(s => s <= slots).ToArray();
var provider = ProviderRegistry.Discover().First(x => x.Availability.IsAvailable && x.Provider.Name != "cpu").Provider;
using var device = provider.OpenDevice(provider.GetDevices()[0], new DeviceOptions());
using var model = LlmModel.Load(argv[0], new LlmLoadOptions { Devices = [new DeviceSlot { Device = device }], ContextLength = ctx, MaxBatchTokens = 512 });
using var dec = model.CreateBatchDecoder(slots, ctx);
Console.WriteLine($"decoder: {dec.GetType().Name}, slots {dec.SlotCount}, batches sequences: {dec.BatchesSequences}, vocab {dec.VocabSize}");
var rng = new Random(1);
int[] Toks(int n) => Enumerable.Range(0, n).Select(_ => rng.Next(1000, 30000)).ToArray();
for (int s = 0; s < slots; s++) dec.Step([new SequenceInput(s, Toks(depth), false)]);   // prefill each slot
double Median(List<double> l) { l.Sort(); return l[l.Count / 2]; }
var segs = () => ((dynamic)typeof(MultiSequenceDecoder).GetProperty("Compiled", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(dec)!).Segments as System.Collections.IEnumerable;
Console.WriteLine("| batch | step ms min | step ms median | tok/s (median) | no-logits ms median |");
Console.WriteLine("|---|---|---|---|---|");
foreach (int b in sizes)
{
    var inputs = Enumerable.Range(0, b).Select(s => new SequenceInput(s, new[] { rng.Next(1000, 30000) }, true)).ToList();
    var noLogits = inputs.Select(i => i with { WantLogits = false }).ToList();
    var t = new List<double>(); var t2 = new List<double>();
    for (int i = 0; i < 8; i++) dec.Step(inputs);
    for (int i = 0; i < 30; i++) { var sw = Stopwatch.StartNew(); dec.Step(inputs); t.Add(sw.Elapsed.TotalMilliseconds); }
    for (int i = 0; i < 30; i++) { var sw = Stopwatch.StartNew(); dec.Step(noLogits); t2.Add(sw.Elapsed.TotalMilliseconds); }
    double med = Median(t);
    Console.WriteLine($"| {b} | {t.Min():F2} | {med:F2} | {b * 1000 / med:F0} | {Median(t2):F2} |");
    if (Opt("--profile") == b.ToString())
    {
        foreach (VulkanSegment seg in segs()!.OfType<VulkanSegment>()) { seg.Profile = true; seg.ResetProfile(); }
        for (int i = 0; i < 20; i++) dec.Step(inputs);
        foreach (VulkanSegment seg in segs()!.OfType<VulkanSegment>()) { Console.WriteLine($"--- profile batch {b}\n{seg.ProfileReport()}"); seg.Profile = false; }
    }
}
