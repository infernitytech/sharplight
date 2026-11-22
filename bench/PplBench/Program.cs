// PplBench <model> --text file.txt [--provider cuda|vulkan] [--seq 2048] [--chunk 2048] [--tokens 8192] [--dump out.f16 [--dump-seqs 1]]
//          [--ids out.ids]
// PplBench --compare a.f16 b.f16
// Perplexity of a model on a fixed text through the prompt path: the text's tokens (no special tokens) are cut into sequences of --seq
// tokens (up to --tokens in all), each a fresh context, fed in steps of --chunk rows with logits for every row (SequenceInput.LogitsRows),
// so every row is computed as a prompt row (the prompt-sized GEMMs). Prints the perplexity over all predicted tokens (each sequence's
// tokens 2..seq), the mean negative log-likelihood, and per sequence. --dump stores the f16 logits of every row of the first
// --dump-seqs sequences; --compare reads two such dumps and prints the mean and max KL(a || b) of the softmaxes, the top-1 agreement
// and the largest logit difference (docs/benchmarks.md section 86). --ids writes the token ids (one line, comma separated), so other
// engines (vLLM's prompt_logprobs) can score the same tokens.
using System.Runtime.InteropServices;
using Microsoft.Extensions.DependencyInjection;
using Sharplight;
using Sharplight.Llm;
using Sharplight.Llm.Serving;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Cuda;

var argv = args.ToList();
string? Opt(string n, string? d = null)
{
    int i = argv.IndexOf(n);
    return i >= 0 && i + 1 < argv.Count ? argv[i + 1] : d;
}

if (Opt("--compare") is { } first)
{
    Compare(first, argv[argv.IndexOf("--compare") + 2]);
    return;
}

int seq = int.Parse(Opt("--seq", "2048")!);
int chunk = int.Parse(Opt("--chunk", seq.ToString())!);
int maxTokens = int.Parse(Opt("--tokens", "8192")!);
using var services = SharplightDefaults.CreateServiceProvider(s => s.AddCudaProvider());
var loader = services.GetRequiredService<ModelLoader>();
string? providerName = Opt("--provider");
var provider = services.GetRequiredService<ExecutionProviderCatalog>().Discover()
    .First(x => x.Availability.IsAvailable && x.Provider.Name != "cpu" && (providerName is null || x.Provider.Name == providerName)).Provider;
using var device = provider.OpenDevice(provider.GetDevices()[0], new DeviceOptions());
using var model = LlmModel.Load(loader, argv[0], new LlmLoadOptions
{
    Devices = [new DeviceSlot { Device = device }], ContextLength = seq, MaxBatchTokens = chunk,
});
using var dec = model.CreateBatchDecoder(1, seq, maxLogitsRows: chunk, kvCacheTokens: 0);
int vocab = dec.VocabSize;
string text = File.ReadAllText(Opt("--text")!);
var all = model.Tokenizer.Encode(text, addSpecial: false, parseSpecial: false).Take(maxTokens).ToArray();
if (Opt("--ids") is { } idsPath)
{
    File.WriteAllText(idsPath, string.Join(",", all));
}

int sequences = all.Length / seq;
string? dumpPath = Opt("--dump");
int dumpSeqs = int.Parse(Opt("--dump-seqs", "1")!);
using var dump = dumpPath is null ? null : File.Create(dumpPath);
var row16 = new Half[vocab];
double nllAll = 0;
long countAll = 0;
var argmax = new List<int>();
for (int s = 0; s < sequences; s++)
{
    var ids = all.AsSpan(s * seq, seq).ToArray();
    dec.Reset(0);
    double nll = 0;
    int count = 0;
    for (int at = 0; at < seq; at += chunk)
    {
        int n = Math.Min(chunk, seq - at);
        var logits = dec.Step([new SequenceInput(0, ids.AsMemory(at, n), true, LogitsRows: n)])[0].Span;
        for (int r = 0; r < n; r++)
        {
            var row = logits.Slice(r * vocab, vocab);
            int p = at + r;
            float max = float.NegativeInfinity;
            int best = 0;
            for (int v = 0; v < vocab; v++)
            {
                if (row[v] > max)
                {
                    max = row[v];
                    best = v;
                }
            }

            argmax.Add(best);
            if (p + 1 < seq)
            {
                double sum = 0;
                for (int v = 0; v < vocab; v++)
                {
                    sum += Math.Exp(row[v] - max);
                }

                nll += max + Math.Log(sum) - row[ids[p + 1]];
                count++;
            }

            if (dump is not null && s < dumpSeqs)
            {
                for (int v = 0; v < vocab; v++)
                {
                    row16[v] = (Half)row[v];
                }

                dump.Write(MemoryMarshal.AsBytes<Half>(row16));
            }
        }
    }

    Console.WriteLine($"sequence {s}: {count} tokens, mean NLL {nll / count:F5}, PPL {Math.Exp(nll / count):F4}");
    nllAll += nll;
    countAll += count;
}

Console.WriteLine($"{sequences} sequences of {seq} tokens ({all.Length} tokens of text), steps of {chunk} rows: {countAll} predictions, "
    + $"mean NLL {nllAll / countAll:F5}, PPL {Math.Exp(nllAll / countAll):F4}");
Console.WriteLine("argmax hash " + argmax.Aggregate(17L, (h, x) => h * 31 + x).ToString("x"));

void Compare(string a, string b)
{
    var fa = File.ReadAllBytes(a);
    var fb = File.ReadAllBytes(b);
    var ha = MemoryMarshal.Cast<byte, Half>(fa);
    var hb = MemoryMarshal.Cast<byte, Half>(fb);
    int v = int.Parse(Opt("--vocab", "151936")!);
    int rows = Math.Min(ha.Length, hb.Length) / v;
    double klSum = 0, klMax = 0, diffMax = 0;
    int same = 0;
    var pa = new double[v];
    var pb = new double[v];
    for (int r = 0; r < rows; r++)
    {
        var ra = ha.Slice(r * v, v);
        var rb = hb.Slice(r * v, v);
        int ia = Softmax(ra, pa), ib = Softmax(rb, pb);
        same += ia == ib ? 1 : 0;
        double kl = 0;
        for (int i = 0; i < v; i++)
        {
            if (pa[i] > 0)
            {
                kl += pa[i] * (Math.Log(pa[i]) - Math.Log(Math.Max(pb[i], 1e-300)));
            }

            diffMax = Math.Max(diffMax, Math.Abs((double)ra[i] - (double)rb[i]));
        }

        klSum += kl;
        klMax = Math.Max(klMax, kl);
    }

    Console.WriteLine($"{rows} rows: KL mean {klSum / rows:E3} max {klMax:E3}, top-1 agreement {same}/{rows} ({100.0 * same / rows:F2} %), max |logit diff| {diffMax:F4}");
}

static int Softmax(ReadOnlySpan<Half> row, double[] p)
{
    double max = double.NegativeInfinity;
    int best = 0;
    for (int i = 0; i < row.Length; i++)
    {
        if ((double)row[i] > max)
        {
            max = (double)row[i];
            best = i;
        }
    }

    double sum = 0;
    for (int i = 0; i < row.Length; i++)
    {
        p[i] = Math.Exp((double)row[i] - max);
        sum += p[i];
    }

    for (int i = 0; i < row.Length; i++)
    {
        p[i] /= sum;
    }

    return best;
}
