using Microsoft.Extensions.DependencyInjection;
using Sharplight.Cpu;
using Sharplight.Distributed.Worker;
using Sharplight.Llm;
using Sharplight.Llm.Serving;
using Sharplight.Models;
using Sharplight.Runtime;

namespace Sharplight.Distributed.Tests;

/// <summary>Models of the local test mirror (<c>~/models/gguf</c>), the CPU device and in-process segment workers on loopback.</summary>
internal static class TestModels
{
    private static readonly string Gguf = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models/gguf");
    private static readonly Lazy<ServiceProvider> s_services = new(() => SharplightDefaults.CreateServiceProvider());
    private static readonly CpuExecutionProvider s_cpu = new();

    /// <summary>A dense model (30 layers).</summary>
    public static string SmolLm => Path.Combine(Gguf, "SmolLM2-135M/SmolLM2-135M.Q8_0.gguf");

    /// <summary>A tiny mixture-of-experts model with latent attention.</summary>
    public static string TinyDeepSeek => Path.Combine(Gguf, "tiny-random-deepseek-f32.gguf");

    /// <summary>A tiny hybrid model: recurrent (gated delta net) and attention layers, routed experts.</summary>
    public static string TinyHybrid => Path.Combine(Gguf, "tiny-random-qwen3.8-flash-next-nomtp-f32.gguf");

    /// <summary>Memory every test device plans with (the host's free memory varies with the other processes on the machine).</summary>
    public const long MemoryLimit = 4L << 30;

    public static ModelLoader Loader => s_services.Value.GetRequiredService<ModelLoader>();

    /// <summary>Skips the test when the model is not in the mirror.</summary>
    public static string Require(string path)
    {
        Assert.SkipUnless(File.Exists(path), "model not found: " + path);
        return path;
    }

    /// <summary>A CPU device (every call opens another; the threads are shared).</summary>
    public static IComputeDevice Cpu() => s_cpu.OpenDevice(s_cpu.GetDevices()[0]);

    /// <summary>A segment worker on a free loopback port with a CPU device and its own copy of the model.</summary>
    public static SegmentWorker StartWorker(string model, Action<SegmentWorkerOptions>? configure = null, ModelDefinition? definition = null)
    {
        var options = new SegmentWorkerOptions
        {
            Listen = "127.0.0.1:0",
            Name = "test-worker",
            Devices = [new DeviceSlot { Device = Cpu(), MemoryLimit = MemoryLimit }],
            ReservePerDevice = 0,
        };
        configure?.Invoke(options);
        var worker = new SegmentWorker(definition ?? Loader.Build(model), options);
        worker.Start();
        return worker;
    }

    /// <summary>The coordinator's options for workers on loopback.</summary>
    public static RemoteSegmentOptions Options(params SegmentWorker[] workers) => new()
    {
        Workers = [.. workers.Select(w => w.Endpoint.ToString())],
        HeartbeatInterval = TimeSpan.FromMilliseconds(200),
        HeartbeatTimeout = TimeSpan.FromSeconds(3),
        ExecuteTimeout = TimeSpan.FromSeconds(60),
    };

    /// <summary>Loads a model over the given device slots (equal layer shares).</summary>
    public static LlmModel Load(string path, IEnumerable<IComputeDevice> devices, int context = 256)
        => LlmModel.Load(Loader, path, new LlmLoadOptions
        {
            Devices = [.. devices.Select(d => new DeviceSlot { Device = d, Weight = 1, MemoryLimit = MemoryLimit })],
            ReservePerDevice = 0,
            ContextLength = context,
            MaxBatchTokens = 32,
        });

    /// <summary>Greedy tokens of one sequence through a batch decoder, and the logits of each step.</summary>
    public static (int[] Tokens, float[][] Logits) Greedy(IBatchDecoder decoder, LlmModel model, string prompt, int count, int slot = 0)
    {
        decoder.Reset(slot);
        var input = model.Tokenizer.Encode(prompt, addSpecial: true, parseSpecial: false);
        var tokens = new List<int>();
        var logits = new List<float[]>();
        for (int i = 0; i < count; i++)
        {
            var row = decoder.Step([new SequenceInput(slot, input)])[0].Span;
            logits.Add(row.ToArray());
            int best = ArgMax(row);
            tokens.Add(best);
            input = [best];
        }

        return ([.. tokens], [.. logits]);
    }

    public static int ArgMax(ReadOnlySpan<float> row)
    {
        int best = 0;
        for (int k = 1; k < row.Length; k++)
        {
            if (row[k] > row[best])
            {
                best = k;
            }
        }

        return best;
    }

    public static float MaxDifference(float[][] a, float[][] b)
    {
        float worst = 0;
        for (int i = 0; i < a.Length; i++)
        {
            for (int k = 0; k < a[i].Length; k++)
            {
                worst = Math.Max(worst, Math.Abs(a[i][k] - b[i][k]));
            }
        }

        return worst;
    }
}
