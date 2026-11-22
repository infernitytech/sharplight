// PipelineBench <model> [--workers h:p,h:p] [--local-stages N] [--provider cpu|cuda|vulkan] [--remote-only] [--slots 8] [--ctx 512]
//               [--depth 64] [--sizes 1,4,8] [--groups 1,2] [--reps 20] [--prefill 512] [--chunks 512,128] [--tokens 16]
// Stages: the local device (or --local-stages copies of it: the same split in one process), then the segment workers.
// 1. greedy tokens of one sequence (compare runs for correctness);
// 2. decode: for each micro-batch count G and batch size B, G multi-sequence decoders (B/G rows each) step concurrently from G threads
//    through the same stages, so one group's step runs on one stage while another's runs on the next; aggregate tokens/s;
// 3. prefill of --prefill tokens in chunks (no logits until the last chunk: consecutive chunks overlap on the stages);
// 4. per worker: executions, round trip, worker time and what the network and protocol added per execution.
using System.Diagnostics;
using System.Globalization;
using Microsoft.Extensions.DependencyInjection;
using Sharplight;
using Sharplight.Distributed;
using Sharplight.Distributed.Transport;
using Sharplight.Graph;
using Sharplight.Llm;
using Sharplight.Llm.Serving;
using Sharplight.Models;
using Sharplight.Runtime;

var argv = args.ToList();
if (argv.Count > 1 && argv[0] == "--echo")
{
    // Echo server for --pingpong: returns every frame.
    using var listener = new TcpSegmentTransport().Listen(SegmentEndpoint.Parse(argv[1]));
    Console.WriteLine($"echo on {listener.LocalEndpoint}");
    while (true)
    {
        var connection = await listener.AcceptAsync(CancellationToken.None);
        _ = Task.Run(() =>
        {
            try
            {
                while (true)
                {
                    var frame = connection.Receive();
                    connection.Send(frame.Kind, frame.RequestId, frame.Payload.Span);
                }
            }
            catch (IOException)
            {
                connection.Dispose();
            }
        });
    }
}

if (argv.Count > 1 && argv[0] == "--pingpong")
{
    // Round trips of one frame of each size to an echo server (another process): the transport's floor per hop.
    using var connection = await new TcpSegmentTransport().ConnectAsync(SegmentEndpoint.Parse(argv[1]), CancellationToken.None);
    foreach (int size in new[] { 64, 16 << 10, 64 << 10, 1 << 20, 8 << 20 })
    {
        var payload = new byte[size];
        int count = size <= 64 << 10 ? 2000 : size <= 1 << 20 ? 200 : 40;
        var times = new List<double>();
        for (int i = 0; i < count + 20; i++)
        {
            long t = Stopwatch.GetTimestamp();
            connection.Send(FrameKind.Execute, (uint)i, payload);
            connection.Receive();
            if (i >= 20)
            {
                times.Add(Stopwatch.GetElapsedTime(t).TotalMilliseconds);
            }
        }

        times.Sort();
        double median = times[times.Count / 2];
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"pingpong {size,9} B: median {median * 1000:F1} us, p99 {times[(int)(times.Count * 0.99)] * 1000:F1} us, {2.0 * size / (median / 1000) / 1e9:F2} GB/s both ways"));
    }

    return;
}
string? Opt(string n, string? d = null)
{
    int i = argv.IndexOf(n);
    return i >= 0 && i + 1 < argv.Count ? argv[i + 1] : d;
}

int[] Ints(string n, string d) => Opt(n, d)!.Split(',').Select(int.Parse).ToArray();

using var services = SharplightDefaults.CreateServiceProvider(s => s.AddCudaProvider());
var loader = services.GetRequiredService<ModelLoader>();
string providerName = Opt("--provider", "cpu")!;
var provider = services.GetRequiredService<ExecutionProviderCatalog>().Discover().First(x => x.Availability.IsAvailable && x.Provider.Name == providerName).Provider;
int localStages = argv.Contains("--remote-only") ? 0 : int.Parse(Opt("--local-stages", "1")!);
var local = Enumerable.Range(0, localStages).Select(_ => provider.OpenDevice(provider.GetDevices()[int.Parse(Opt("--device", "0")!)])).ToList();
var workers = (Opt("--workers") ?? "").Split(',', StringSplitOptions.RemoveEmptyEntries);
using var remoteProvider = new RemoteExecutionProvider(new RemoteSegmentOptions { Workers = workers });
var remote = workers.Length > 0 ? remoteProvider.OpenAll() : [];
var devices = local.Cast<IComputeDevice>().Concat(remote).ToList();
int slots = int.Parse(Opt("--slots", "8")!);
int ctx = int.Parse(Opt("--ctx", "512")!);
int depth = int.Parse(Opt("--depth", "64")!);
int reps = int.Parse(Opt("--reps", "20")!);
using var model = LlmModel.Load(loader, argv[0], new LlmLoadOptions
{
    Devices = [.. devices.Select(d => new DeviceSlot { Device = d, Weight = 1 })],
    ContextLength = ctx,
    MaxBatchTokens = int.Parse(Opt("--batch", "512")!),
});
Console.WriteLine($"stages: {string.Join(" | ", devices.Select(d => d.Info.Name))}");

var rng = new Random(1);
int[] Toks(int n) => Enumerable.Range(0, n).Select(_ => rng.Next(1000, 30000)).ToArray();
double Median(List<double> l)
{
    l.Sort();
    return l[l.Count / 2];
}

void Reset()
{
    foreach (var d in remote)
    {
        d.ResetStatistics();
    }
}

void Report(string phase)
{
    foreach (var d in remote)
    {
        var s = d.Statistics;
        if (s.Executions > 0)
        {
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"  [{phase}] {d.Endpoint}: {s.Executions} exec, round trip {s.RoundTripMs / s.Executions:F3} ms, worker {s.WorkerMs / s.Executions:F3} ms, wait {s.WaitMs / s.Executions:F3} ms, overhead {s.OverheadMsPerExecution:F3} ms, "
                + $"{s.BytesSent / (double)s.Executions / 1024:F1} KiB out, {s.BytesReceived / (double)s.Executions / 1024:F1} KiB in per exec"));
        }
    }
}

// Device sampling (a sampling head on the last stage) where the devices compiled one; otherwise logits come back to the host.
bool deviceSampling;
using (var probe = (MultiSequenceDecoder)model.CreateBatchDecoder(2, ctx))
{
    try
    {
        probe.Step([new SequenceInput(0, new[] { 1 })], [SamplingRow.Greedy()], new int[1], new float[1]);
        deviceSampling = true;
    }
    catch (InvalidOperationException)
    {
        deviceSampling = false;
    }
}

Console.WriteLine($"sampling: {(deviceSampling ? "on the last stage (token ids return)" : "on the host (logits return)")}");

// 1. Greedy tokens of one sequence.
using (var decoder = (MultiSequenceDecoder)model.CreateBatchDecoder(2, ctx))
{
    var input = model.Tokenizer.Encode("The capital of France is", addSpecial: true, parseSpecial: false);
    var chosen = new List<int>();
    var tokens = new int[1];
    var logprobs = new float[1];
    for (int i = 0; i < int.Parse(Opt("--tokens", "16")!); i++)
    {
        int next = deviceSampling
            ? Sample(decoder, [new SequenceInput(0, input)], tokens, logprobs)
            : Sampler.ArgMax(decoder.Step([new SequenceInput(0, input)])[0].Span);
        chosen.Add(next);
        tokens[0] = next;
        input = [tokens[0]];
    }

    Console.WriteLine($"greedy: {string.Join(',', chosen)} \"{model.Tokenizer.Decode(chosen.ToArray())}\"");
}

// 2. Decode with G concurrent micro-batches.
foreach (int groups in Ints("--groups", "1,2"))
{
    var decoders = Enumerable.Range(0, groups).Select(_ => (MultiSequenceDecoder)model.CreateBatchDecoder(Math.Max(2, slots / groups), ctx)).ToList();
    foreach (var decoder in decoders)
    {
        for (int s = 0; s < decoder.SlotCount; s++)
        {
            decoder.Step([new SequenceInput(s, Toks(depth), false)]);
        }
    }

    foreach (int size in Ints("--sizes", "1,4,8").Where(b => b >= groups && b <= slots))
    {
        int per = size / groups;
        var steps = new List<double>();
        Reset();
        for (int rep = 0; rep < reps + 2; rep++)
        {
            var watch = Stopwatch.StartNew();
            Parallel.For(0, groups, new ParallelOptions { MaxDegreeOfParallelism = groups }, g =>
            {
                var decoder = decoders[g];
                var inputs = Enumerable.Range(0, per).Select(s => new SequenceInput(s, new[] { rng.Next(1000, 30000) })).ToList();
                var tokens = new int[per];
                var logprobs = new float[per];
                if (deviceSampling)
                {
                    decoder.Step(inputs, [.. inputs.Select(_ => (SamplingRow?)SamplingRow.Greedy())], tokens, logprobs);
                }
                else
                {
                    foreach (var logits in decoder.Step(inputs))
                    {
                        Sampler.ArgMax(logits.Span);
                    }
                }
            });
            if (rep >= 2)
            {
                steps.Add(watch.Elapsed.TotalMilliseconds);
            }

            foreach (var decoder in decoders)
            {
                for (int s = 0; s < per; s++)
                {
                    if (decoder.Length(s) >= ctx - 2)
                    {
                        decoder.Truncate(s, depth);
                    }
                }
            }
        }

        double median = Median(steps);
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"decode G={groups} B={size}: median {median:F2} ms/step, min {steps.Min():F2} ms, {per * groups * 1000 / median:F1} tok/s"));
        Report($"decode G={groups} B={size}");
    }

    foreach (var decoder in decoders)
    {
        decoder.Dispose();
    }
}

// 3. Prefill in chunks.
int prefill = int.Parse(Opt("--prefill", "512")!);
using (var decoder = (MultiSequenceDecoder)model.CreateBatchDecoder(2, Math.Max(ctx, prefill + 8)))
{
    var prompt = Toks(prefill);
    foreach (int chunk in Ints("--chunks", "512,128"))
    {
        var times = new List<double>();
        Reset();
        for (int rep = 0; rep < 4; rep++)
        {
            decoder.Reset(0);
            var watch = Stopwatch.StartNew();
            for (int done = 0; done < prefill; done += chunk)
            {
                int n = Math.Min(chunk, prefill - done);
                bool last = done + n >= prefill;
                decoder.Step([new SequenceInput(0, prompt.AsMemory(done, n), last)]);
            }

            if (rep > 0)
            {
                times.Add(watch.Elapsed.TotalMilliseconds);
            }
        }

        double median = Median(times);
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture, $"prefill {prefill} tok in chunks of {chunk}: median {median:F1} ms, {prefill * 1000 / median:F1} tok/s"));
        Report($"prefill chunk {chunk}");
    }
}

foreach (var d in devices)
{
    d.Dispose();
}

static int Sample(MultiSequenceDecoder decoder, List<SequenceInput> inputs, int[] tokens, float[] logprobs)
{
    decoder.Step(inputs, [SamplingRow.Greedy()], tokens, logprobs);
    return tokens[0];
}
