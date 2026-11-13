using System.Diagnostics;
using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Cpu;
using Sharplight.Gguf;
using Sharplight.HuggingFace;
using Sharplight.Package;
using Sharplight.Vision;
using Sharplight.Vulkan;
using StbImageSharp;

// sharplight <model.gguf | model.slm | hf-checkpoint-dir> [--mmproj vision.gguf] [--image img.png ...] [--image-max-tokens n] [--save out.slm] [-p prompt] [-n tokens] [--temp t] [--ctx n] [--batch n] [--devices] [--split a,b] [--gpu-mem GiB] [--cpu-offload] [--disable-features a,b] [--bench N]
var argv = args.ToList();
string? Opt(string name, string? def = null) { int i = argv.IndexOf(name); return i >= 0 && i + 1 < argv.Count ? argv[i + 1] : def; }
bool Flag(string name) => argv.Contains(name);
List<string> Opts(string name) { var l = new List<string>(); for (int i = 0; i + 1 < argv.Count; i++) if (argv[i] == name) l.Add(argv[i + 1]); return l; }

ProviderRegistry.Register(() => new VulkanProvider());
CpuExecutionProvider.Register();
GgufImporter.Register();
HuggingFaceImporter.Register();
ModelPackageImporter.Register();
if (Flag("--devices") || argv.Count == 0)
{
    foreach (var (p, avail) in ProviderRegistry.Discover())
    {
        Console.WriteLine($"{p.Name}: {(avail.IsAvailable ? "available" : "unavailable")} — {avail.Reason}");
        if (avail.IsAvailable)
            foreach (var d in p.GetDevices())
                Console.WriteLine($"  [{d.Index}] {d}\n      features: {string.Join(" ", d.Features.Order())}\n      " + string.Join(", ", d.Capabilities.Select(kv => $"{kv.Key}={kv.Value}")));
    }
    if (argv.Count <= 1) return;
}

var modelPath = argv[0];
// --cpu: run everything on the CPU provider instead of a GPU.
var provider = ProviderRegistry.Discover().First(x => x.Availability.IsAvailable && (x.Provider.Name == "cpu") == Flag("--cpu")).Provider;
var devInfo = provider.GetDevices()[int.Parse(Opt("--device", "0")!)];
// --split 0.5,0.5 opens the same GPU several times to exercise the multi-device pipeline path.
var split = Opt("--split")?.Split(',').Select(double.Parse).ToArray() ?? [1.0];
long? gpuMem = Opt("--gpu-mem") is { } gm ? (long)(double.Parse(gm, System.Globalization.CultureInfo.InvariantCulture) * (1L << 30)) : null;
// --disable-features matrix_f16,fp16,...: never use these device features (see DeviceFeatures).
var devOptions = new DeviceOptions();
foreach (var f in (Opt("--disable-features") ?? "").Split(',', StringSplitOptions.RemoveEmptyEntries)) devOptions.DisabledFeatures.Add(f.Trim());
var slots = split.Select(w => new DeviceSlot { Device = provider.OpenDevice(devInfo, devOptions), Weight = w, MemoryLimit = gpuMem }).ToList();
// --cpu-offload: layers that don't fit (see --gpu-mem) and unsupported ops run on the CPU.
DeviceSlot? fallback = null;
if (Flag("--cpu-offload"))
{
    var cpu = new CpuExecutionProvider();
    fallback = new DeviceSlot { Device = cpu.OpenDevice(cpu.GetDevices()[0]) };
}

// --image path (repeatable): the prompt is formatted with the model's chat template and the images are encoded by the
// model's vision tower (HF checkpoints contain it; GGUF models need --mmproj vision.gguf).
var imagePaths = Opts("--image");
var imageTokens = Opt("--image-max-tokens") is not null || Opt("--image-min-tokens") is not null
    ? new ImageTokenLimits(Opt("--image-min-tokens") is { } imn ? int.Parse(imn) : null, Opt("--image-max-tokens") is { } imx ? int.Parse(imx) : null) : null;
using var model = LlmModel.Load(modelPath, new LlmLoadOptions
{
    Auxiliary = Opt("--mmproj") is { } mmproj ? [("mmproj", mmproj)] : [],
    ImageTokens = imageTokens,
    Devices = slots,
    Fallback = fallback,
    ContextLength = int.Parse(Opt("--ctx", "4096")!),
    MaxBatchTokens = int.Parse(Opt("--batch", "256")!),
    Log = Console.Error.WriteLine,
});
var session = model.CreateSession(imagePaths.Count > 0 ? WellKnown.DecodeMultimodalGraph : WellKnown.DecodeGraph);
// --save out.slm: store the built model and this device's compiled kernels + pipeline cache in one file. When the
// model already is that package, the compiled target is appended to it instead.
if (Opt("--save") is { } savePath)
{
    var swSave = Stopwatch.StartNew();
    var targets = model.ExportCompilation();
    if (model.Source.Path is { } src && ModelPackage.IsPackage(src) && Path.GetFullPath(src) == Path.GetFullPath(savePath))
        ModelPackage.AddTargets(savePath, targets);
    else
        ModelPackage.Save(savePath, model.Definition, targets, Console.Error.WriteLine);
    Console.Error.WriteLine($"Saved {savePath}: {new FileInfo(savePath).Length / 1e9:F2} GB, targets: {string.Join("; ", targets.Select(t => $"{t.Target} ({t.Entries.Count} entries)"))} [{swSave.ElapsedMilliseconds} ms]");
}
if (Opt("--bench") is { } benchArg)
{
    // --bench N: warm prefill of N synthetic tokens, then 64 decode steps. Profile with SHARPLIGHT_VK_PROFILE=1.
    int nb = int.Parse(benchArg);
    var rng = new Random(1);
    var toks = Enumerable.Range(0, nb).Select(_ => rng.Next(1000, 100000)).ToArray();
    void ResetProfiles() { foreach (var seg in session.Compiled.Segments.OfType<VulkanSegment>()) seg.ResetProfile(); }
    void PrintProfiles(string what) { foreach (var seg in session.Compiled.Segments.OfType<VulkanSegment>()) if (seg.Profile) Console.Error.WriteLine($"--- {what} profile, segment {seg.Segment.Index}\n{seg.ProfileReport()}"); }
    session.Process(toks); session.Process([5]); session.Reset(); // warm-up: record command buffers
    foreach (var seg in session.Compiled.Segments.OfType<VulkanSegment>())
    {
        Console.Error.WriteLine($"segment {seg.Segment.Index}: {seg.Stats(1).Dispatches} dispatches, {seg.Stats(1).Barriers} barriers per step");
        if (Environment.GetEnvironmentVariable("SHARPLIGHT_DUMP_SCHEDULE") == "1")
            foreach (var (name, b) in seg.Schedule()) Console.Error.WriteLine($"{(b ? "--- barrier" + Environment.NewLine : "")}    {name}");
    }
    ResetProfiles();
    var sw0 = Stopwatch.StartNew();
    session.Process(toks);
    double pre = sw0.Elapsed.TotalSeconds;
    PrintProfiles($"prefill T={nb}");
    ResetProfiles();
    var steps = new List<double>();
    double best = pre;
    for (int rep = 0; rep < 3; rep++) { session.Reset(); sw0.Restart(); session.Process(toks); best = Math.Min(best, sw0.Elapsed.TotalSeconds); }
    session.Reset(); session.Process(toks);
    ResetProfiles();
    var gpu = new List<double>();
    for (int i = 0; i < 64; i++)
    {
        sw0.Restart(); session.Process([toks[i % nb]]); steps.Add(sw0.Elapsed.TotalSeconds);
        gpu.Add(session.Compiled.Segments.OfType<VulkanSegment>().Sum(s => s.LastGpuMs));
    }
    gpu.Sort();
    double dec = steps.Sum();
    steps.Sort();
    PrintProfiles("decode x64");
    // Best/median are robust against other GPU users; the mean is what a user would see.
    Console.WriteLine($"prefill {nb} tok: best {best * 1000:F0} ms = {nb / best:F1} tok/s | decode: best {1 / steps[0]:F2}, median {1 / steps[steps.Count / 2]:F2}, mean {64 / dec:F2} tok/s | GPU ms/step: best {gpu[0]:F2}, median {gpu[gpu.Count / 2]:F2}, wall best {steps[0] * 1000:F2}");
    session.Dispose();
    foreach (var d in slots) d.Device.Dispose();
    fallback?.Device.Dispose();
    return;
}
VisionEncoder? vision = null;
var images = new List<ImageInput>();
if (imagePaths.Count > 0)
{
    vision = model.CreateVisionEncoder();
    foreach (var path in imagePaths)
    {
        var img = ImageResult.FromMemory(File.ReadAllBytes(path), ColorComponents.RedGreenBlue);
        images.Add(new ImageInput(img.Width, img.Height, img.Data));
    }
}
var gen = new Generator(session, model.Tokenizer, vision);
var prompt = Opt("-p", "The capital of France is")!.Replace("\\n", "\n");
// --logits out.bin: process the prompt, print token ids and the top-5 logits, store the last-token logits as raw float32.
if (Opt("--logits") is { } logitsPath)
{
    var ids = model.Tokenizer.Encode(prompt, addSpecial: true, parseSpecial: true);
    var lg = session.Process(ids);
    File.WriteAllBytes(logitsPath, System.Runtime.InteropServices.MemoryMarshal.AsBytes<float>(lg).ToArray());
    Console.WriteLine("ids: " + string.Join(",", ids));
    foreach (var i in Enumerable.Range(0, lg.Length).OrderByDescending(i => lg[i]).Take(5)) Console.WriteLine($"  {i} {lg[i]:F4} {model.Tokenizer.Decode([i])}");
    session.Dispose();
    foreach (var d in slots) d.Device.Dispose();
    fallback?.Device.Dispose();
    return;
}
if (images.Count > 0)
{
    // Chat formatting: the template turns each image part into the model's placeholder tokens.
    var template = new Sharplight.Templating.ChatTemplate(model.ChatTemplate ?? throw new InvalidOperationException("The model has no chat template."), "", "");
    var parts = images.Select(_ => Sharplight.Templating.ChatContentPart.Image()).Append(Sharplight.Templating.ChatContentPart.FromText(prompt)).ToList();
    prompt = template.Render([new Sharplight.Templating.ChatMessage("user", Parts: parts)], addGenerationPrompt: true,
        new Dictionary<string, object?> { ["enable_thinking"] = Flag("--think") });
}
float temp = float.Parse(Opt("--temp", "0")!, System.Globalization.CultureInfo.InvariantCulture);
var options = new GenerationOptions
{
    MaxNewTokens = int.Parse(Opt("-n", "64")!),
    Sampling = temp <= 0 ? SamplingOptions.Greedy : new SamplingOptions { Temperature = temp, Seed = 42 },
};
// --chat: wrap the prompt as a user message and render it with the model's chat template (the template emits BOS itself).
int[]? promptIds = null;
if (Flag("--chat") && images.Count == 0)
{
    if (model.ChatTemplate is not { } tpl) throw new InvalidOperationException("The model has no chat template.");
    string? Tok(int? id) => id is int i ? model.Tokenizer.TokenString(i) : null;
    var rendered = new Sharplight.Templating.ChatTemplate(tpl, Tok(model.Tokenizer.BosId), Tok(model.Tokenizer.EosId))
        .Render([new Sharplight.Templating.ChatMessage("user", prompt)], addGenerationPrompt: true);
    Console.WriteLine("--- rendered prompt ---\n" + rendered + "\n-----------------------");
    promptIds = model.Tokenizer.Encode(rendered, addSpecial: false, parseSpecial: true);
    prompt = "";
}
if (images.Count == 0) Console.Write(prompt);
foreach (var t in images.Count > 0 ? gen.Generate(prompt, images, options) : promptIds is null ? gen.Generate(prompt, options) : gen.Generate(promptIds, options)) Console.Write(t.Text);
Console.WriteLine();
var s = gen.LastStats!;
Console.Error.WriteLine($"prompt: {s.PromptTokens} tok in {s.PrefillTime.TotalMilliseconds:F0} ms ({s.PrefillTokensPerSecond:F1} tok/s) | generated: {s.GeneratedTokens} tok, {s.DecodeTokensPerSecond:F2} tok/s");
if (images.Count > 0) Console.Error.WriteLine($"images: {images.Count} preprocessed and encoded in {gen.LastImageTime.TotalMilliseconds:F0} ms");
vision?.Dispose();
session.Dispose();
foreach (var d in slots) d.Device.Dispose();
fallback?.Device.Dispose();
