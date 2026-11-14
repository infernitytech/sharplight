using System.Text.Json;
using Sharplight.Cpu;
using Sharplight.Gguf;
using Sharplight.HuggingFace;
using Sharplight.Llm;
using Sharplight.Llm.Serving;
using Sharplight.Llm.Serving.Chat;
using Sharplight.Llm.Serving.Hosting;
using Sharplight.Runtime;
using Sharplight.Vulkan;

namespace Sharplight.Families.Tests;

/// <summary>
/// Masked-diffusion models (Dream, LLaDA) behind the continuous-batching engine and the chat-model loader of the serving host: requests of
/// different prompt lengths denoised together (packed into shared passes) produce exactly what <see cref="DiffusionGenerator"/> produces for
/// each alone, and the reference prompt reproduces the transformers reference of <c>tools/mk_tiny_diffusion.py</c>.
/// </summary>
public class DiffusionServingTests
{
    private static readonly string Home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

    static DiffusionServingTests()
    {
        GgufImporter.Register();
        HuggingFaceImporter.Register();
    }

    /// <summary>Batched engine output equals the single-sequence generator, and the reference prompt gives the reference ids.</summary>
    [Theory]
    [InlineData("hf/tiny-random-dream", false)]
    [InlineData("hf/tiny-random-dream", true)]
    [InlineData("hf/tiny-random-llada", false)]
    [InlineData("hf/tiny-random-llada", true)]
    [InlineData("gguf/tiny-random-llada-f32.gguf", false)]
    public async Task Engine_Batched_Matches_Generator(string model, bool vulkan)
    {
        var (path, reference) = Locate(model);
        var (prompt, expected, options) = Reference(reference);
        using var device = OpenDevice(vulkan);
        using var llm = LlmModel.Load(path, new LlmLoadOptions { Devices = [new DeviceSlot { Device = device }], ContextLength = 64, MaxBatchTokens = 64 });
        var rng = new Random(5);
        // 36 + 28 rows share the first pass, the third sequence gets its own.
        int[][] prompts = [prompt, .. new[] { 4, 6 }.Select(n => Enumerable.Range(0, n).Select(_ => rng.Next(10, 900)).ToArray())];
        var single = new int[prompts.Length][];
        using (var generator = llm.CreateDiffusionGenerator())
        {
            for (int i = 0; i < prompts.Length; i++)
            {
                single[i] = generator.Generate(prompts[i], options);
            }
        }

        Assert.Equal(expected, single[0]);
        var ct = TestContext.Current.CancellationToken;
        using var engine = new InferenceEngine(llm.CreateBatchDecoder(slots: prompts.Length), llm.Tokenizer);
        var handles = prompts.Select(p => engine.Submit(new GenerationRequest
        {
            PromptTokens = p, MaxTokens = options.MaxNewTokens, StopAtEog = false, Diffusion = options.Diffusion,
        }, ct)).ToList();
        var results = await Task.WhenAll(handles.Select(h => h.CompleteAsync(ct)));
        for (int i = 0; i < prompts.Length; i++)
        {
            Assert.Null(results[i].Error);
            Assert.Equal(FinishReason.Length, results[i].Finish);
            Assert.Equal(options.MaxNewTokens, results[i].Usage.CompletionTokens);
            Assert.Equal(llm.Tokenizer.Decode(single[i]), results[i].Text);
        }
    }

    /// <summary>The chat-model loader of the serving host loads a diffusion model; completions match the reference.</summary>
    [Fact]
    public async Task Loader_Serves_Completions()
    {
        var (path, reference) = Locate("hf/tiny-random-llada");
        var (prompt, expected, options) = Reference(reference);
        using var device = OpenDevice(false);
        var loader = new LlmChatModelLoader("llada", path, new LlmLoadOptions { Devices = [new DeviceSlot { Device = device }], ContextLength = 64, MaxBatchTokens = 64 },
            new ChatModelOptions { Slots = 2 });
        var ct = TestContext.Current.CancellationToken;
        using var prepared = loader.Prepare(ct);
        Assert.True(prepared.Requirements.Sum(r => r.Bytes) > 0);
        using var loaded = prepared.Load(null);
        var chat = Assert.IsType<ChatModelInstance>(loaded);
        var request = new CompletionRequest { PromptTokens = prompt, MaxTokens = options.MaxNewTokens, IgnoreEos = true, Diffusion = options.Diffusion, N = 2 };
        var result = await chat.Service.StartCompletion(request, ct).CompleteAsync(ct);
        Assert.Equal(2, result.Choices.Count);
        foreach (var choice in result.Choices)
        {
            Assert.Null(choice.Error);
            Assert.Equal(chat.Engine.Tokenizer.Decode(expected), choice.Content);
        }

        // A schedule that does not fit the span is rejected up front.
        var bad = new CompletionRequest { PromptTokens = prompt, MaxTokens = 10, Diffusion = new DiffusionOptions { BlockLength = 4, Steps = 3 } };
        var error = Assert.Throws<EngineException>(() => chat.Service.StartCompletion(bad, ct));
        Assert.Equal("invalid_request", error.Code);
    }

    private static (string Path, string Reference) Locate(string model)
    {
        string path = Path.Combine(Home, "models", model);
        string name = Path.GetFileName(model).Replace("-f32.gguf", "", StringComparison.Ordinal);
        string reference = Path.Combine(Home, "models", "hf", name, "reference_diffusion.json");
        Assert.SkipUnless((File.Exists(path) || Directory.Exists(path)) && File.Exists(reference), $"{path} or its reference not present");
        return (path, reference);
    }

    private static (int[] Prompt, int[] Ids, GenerationOptions Options) Reference(string file)
    {
        using var json = JsonDocument.Parse(File.ReadAllText(file));
        var r = json.RootElement;
        var options = new GenerationOptions
        {
            MaxNewTokens = r.GetProperty("span").GetInt32(),
            StopAtEog = false,
            Diffusion = new DiffusionOptions
            {
                Steps = r.GetProperty("steps").GetInt32(),
                BlockLength = r.GetProperty("block_length").GetInt32(),
                Eps = r.GetProperty("eps").GetSingle(),
            },
        };
        return ([.. r.GetProperty("prompt").EnumerateArray().Select(e => e.GetInt32())], [.. r.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32())], options);
    }

    private static IComputeDevice OpenDevice(bool vulkan)
    {
        if (!vulkan)
        {
            var cpu = new CpuExecutionProvider();
            return cpu.OpenDevice(cpu.GetDevices()[0]);
        }

        var p = new VulkanProvider();
        Assert.SkipUnless(p.CheckAvailability().IsAvailable, "Vulkan is not available");
        var devs = p.GetDevices();
        var dev = devs.FirstOrDefault(d => d.Kind == DeviceKind.DiscreteGpu) ?? devs.FirstOrDefault();
        Assert.SkipWhen(dev is null, "no Vulkan device");
        return p.OpenDevice(dev!);
    }
}
