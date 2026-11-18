using Sharplight.Cuda;
using Sharplight.Llm;
using Sharplight.Runtime;
using Sharplight.Vulkan;

namespace Sharplight.Families.Tests;

/// <summary>
/// Greedy generation on the CUDA provider against the Vulkan provider on the same GPU: both lower the graph with the same rules and run
/// the same Slang kernels, so the tokens must agree. Opt-in with <c>SHARPLIGHT_TEST_CUDA=1</c> (<c>CUDA_VISIBLE_DEVICES</c> and
/// <c>SHARPLIGHT_VK_DEVICES</c> select the GPU); each case skips when its model is absent (under ~/models).
/// </summary>
public class CudaParityTests
{
    private const int Tokens = 32;
    private const string Prompt = "The capital of France is a city that, in the year 1900 and beyond, was known for many things such as art";

    private static readonly string Home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

    public static TheoryData<string> Models => new()
    {
        "gguf/Qwen3-0.6B/Qwen3-0.6B-Q8_0.gguf",
        "gguf/Llama-3.2-1B-Instruct/Llama-3.2-1B-Instruct-Q8_0.gguf",
        "gguf/gemma-3-1b-it/gemma-3-1b-it-Q8_0.gguf",
        "hf/Qwen3-0.6B",
    };

    [Theory, MemberData(nameof(Models))]
    public void GreedyTokensMatchVulkan(string model)
    {
        Assert.SkipUnless(Environment.GetEnvironmentVariable("SHARPLIGHT_TEST_CUDA") == "1", "the CUDA provider runs with SHARPLIGHT_TEST_CUDA=1");
        string path = Path.Combine(Home, "models", model);
        Assert.SkipUnless(File.Exists(path) || Directory.Exists(path), $"{path} not present");
        var cuda = new CudaProvider();
        var vulkan = new VulkanProvider();
        Assert.SkipUnless(cuda.CheckAvailability().IsAvailable && vulkan.CheckAvailability().IsAvailable, "needs the CUDA and the Vulkan provider");

        var expected = Generate(vulkan, path);
        var actual = Generate(cuda, path);

        Assert.Equal(expected, actual);
    }

    private static int[] Generate(IExecutionProvider provider, string path)
    {
        var info = provider.GetDevices().First(d => d.Kind == DeviceKind.DiscreteGpu);
        using var device = provider.OpenDevice(info);
        // The integer-dot matvec quantizes activations; it is off so that the comparison checks the provider, not that approximation.
        using var model = LlmModel.Load(TestHost.Loader, path, new LlmLoadOptions
        {
            Devices = [new DeviceSlot { Device = device }], ContextLength = 256, MaxBatchTokens = 256,
            DisabledFeatures = new HashSet<string> { DeviceFeatures.IntDot },
        });
        using var session = model.CreateSession(WellKnown.DecodeGraph);
        var generator = new Generator(session, model.Tokenizer);
        return [.. generator.Generate(Prompt, new GenerationOptions { MaxNewTokens = Tokens, Sampling = SamplingOptions.Greedy }).Where(t => t.Id >= 0).Select(t => t.Id)];
    }
}
