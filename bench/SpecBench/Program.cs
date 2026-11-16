// SpecBench <model> [--drafts 0,1,2,3] [--method mtp|ngram] [--depths 16,4096] [--reps 5] [--ctx 4608] [--draft file] [--gpus N] [--cpu-offload]
// <model> is a GGUF, a .slm package or a Hugging Face safetensors directory. --draft loads a draft file (an mtp-*.gguf MTP sidecar, an
// EAGLE-3 or DFlash drafter) as the draft source; --gpus N places the model over the first N GPUs; --cpu-offload adds the CPU for what
// does not fit. --dump DIR writes the greedy text of 200 new tokens after the plain essay prompt per draft count (DIR/k{k}.txt), for
// checking that speculative output equals plain decoding through the same engine.
// Single-stream speculative decoding through the InferenceEngine (one slot), mirroring bench/vllm_single.py --text-prompt:
// greedy decoding after a natural-language prompt (padded with random tokens to the depth), end-of-generation ignored;
// tg = 128 / (t(129 new tokens) - t(1 new token)), median / min / max over --reps pairs. Per draft count it also reports the
// draft acceptance (accepted / verified drafts) and the mean accepted length per verification step (1 + accepted / steps).
// Draft count 0 is the plain decoder (no speculator). "prompt + 1 token" is t(1 new token): prefill of the depth plus one step.
using System.Diagnostics;
using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Llm.Serving;
using Sharplight.Llm.Serving.Speculative;
using Sharplight.Runtime;
using Sharplight.Samples;

const string Text = "Write a detailed essay about the history of the printing press and its influence on science, religion and politics in Europe. "
    + "Explain the technical inventions involved, name the most important people and give concrete examples of books that changed the world.\n\n";

var argv = args.ToList();
string? Opt(string n, string? d = null)
{
    int i = argv.IndexOf(n);
    return i >= 0 && i + 1 < argv.Count ? argv[i + 1] : d;
}

using var runtime = new SampleRuntime();
var loader = runtime.Loader;
int ctx = int.Parse(Opt("--ctx", "4608")!);
int reps = int.Parse(Opt("--reps", "5")!);
var drafts = Opt("--drafts", "0,1,2,3")!.Split(',').Select(int.Parse).ToArray();
var depths = Opt("--depths", "16,4096")!.Split(',').Select(int.Parse).ToArray();
string method = Opt("--method", "mtp")!;
int gpus = int.Parse(Opt("--gpus", "1")!);
string? draftFile = Opt("--draft");
var provider = runtime.SelectProvider(cpu: false);
var devices = Enumerable.Range(0, gpus).Select(i => provider.OpenDevice(provider.GetDevices()[i], new DeviceOptions())).ToList();
IComputeDevice? cpuDevice = null;
if (argv.Contains("--cpu-offload"))
{
    var cpuProvider = runtime.SelectProvider(cpu: true);
    cpuDevice = cpuProvider.OpenDevice(cpuProvider.GetDevices()[0], new DeviceOptions());
}

var auxiliary = new Dictionary<string, string>(StringComparer.Ordinal);
if (draftFile is not null)
{
    auxiliary[Sharplight.Vision.AuxiliaryRoles.Draft] = draftFile;
}

using var model = LlmModel.Load(loader, argv[0], new LlmLoadOptions
{
    Devices = [.. devices.Select(d => new DeviceSlot { Device = d })],
    Fallback = cpuDevice is null ? null : new DeviceSlot { Device = cpuDevice },
    ContextLength = ctx,
    MaxBatchTokens = 512,
    Auxiliary = auxiliary,
});
var textIds = model.Tokenizer.Encode(Text, addSpecial: false);
var rng = new Random(1);
int[] Prompt(int n, bool text)
{
    int pad = text ? Math.Max(0, n - textIds.Length) : n;
    int[] p = [.. Enumerable.Range(0, pad).Select(_ => rng.Next(1000, 30000))];
    return text ? [.. p, .. textIds] : p;
}

double Median(List<double> l)
{
    var s = l.Order().ToList();
    return s[s.Count / 2];
}

foreach (int k in drafts)
{
    var decoder = k == 0 ? model.CreateBatchDecoder(1, ctx) : model.CreateBatchDecoder(1, ctx, maxLogitsRows: k + 1, draftTokens: k);
    ISpeculator? spec = k == 0 ? null : method == "mtp" ? model.CreateMtpSpeculator(1, k, ctx) : new NgramSpeculator(k);
    using var engine = new InferenceEngine(decoder, model.Tokenizer, new EngineOptions { Speculator = spec });
    double Timed(int n, int newTokens, bool text)
    {
        var request = new GenerationRequest
        {
            PromptTokens = Prompt(n, text), MaxTokens = newTokens, MinTokens = newTokens, StopAtEog = false, Sampling = SamplingParams.GreedyDefault,
        };
        var sw = Stopwatch.StartNew();
        var r = engine.Submit(request).CompleteAsync().GetAwaiter().GetResult();
        if (r.Error is not null || r.Usage.CompletionTokens != newTokens)
        {
            throw new InvalidOperationException($"generation failed: {r.Error} ({r.Usage.CompletionTokens} tokens)");
        }

        return sw.Elapsed.TotalSeconds;
    }

    for (int i = 0; i < 3; i++)
    {
        Timed(512, 8, false);   // warm-up (command-buffer recording for the shapes)
    }

    foreach (int depth in depths)
    {
        int d = Math.Max(depth, textIds.Length + 1);
        Timed(d, 129, true);
        var m0 = engine.GetMetrics();
        var t1 = new List<double>();
        var t129 = new List<double>();
        for (int i = 0; i < reps; i++)
        {
            t1.Add(Timed(d, 1, true));
            t129.Add(Timed(d, 129, true));
        }

        var m1 = engine.GetMetrics();
        var tg = t1.Zip(t129, (a, b) => 128 / (b - a)).ToList();
        long drafted = m1.DraftTokens - m0.DraftTokens;
        long accepted = m1.AcceptedDraftTokens - m0.AcceptedDraftTokens;
        string acc = drafted > 0 ? $" acceptance {(double)accepted / drafted:F2} ({accepted}/{drafted})" : "";
        Console.WriteLine($"{method} k={k} depth={d}: tg median {Median(tg):F1} min {tg.Min():F1} max {tg.Max():F1} tok/s, prompt + 1 token {Median(t1) * 1000:F0} ms (min {t1.Min() * 1000:F0}){acc}");
    }

    if (Opt("--dump") is { } dumpDir)
    {
        Directory.CreateDirectory(dumpDir);
        var request = new GenerationRequest { PromptTokens = textIds, MaxTokens = 200, MinTokens = 200, StopAtEog = false, Sampling = SamplingParams.GreedyDefault };
        var result = engine.Submit(request).CompleteAsync().GetAwaiter().GetResult();
        File.WriteAllText(Path.Combine(dumpDir, $"k{k}.txt"), result.Text);
    }
}
