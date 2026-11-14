using Sharplight.Cpu;
using Sharplight.Gguf;
using Sharplight.Llm;
using Sharplight.Runtime;

namespace Sharplight.Extensions.AI.Tests;

/// <summary>
/// The chat client over a masked-diffusion model (the tiny random Dream GGUF written by the Families tests' <c>tools/convert_tiny_gguf.py</c>):
/// the backend is the diffusion one and streams what <see cref="DiffusionGenerator"/> produces for the same prompt.
/// </summary>
public class DiffusionBackendTests
{
    private static readonly string Path = System.IO.Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        "models", "gguf", "tiny-random-dream-f32.gguf");

    static DiffusionBackendTests() => GgufImporter.Register();

    [Fact]
    public async Task ChatClientPicksDiffusionBackendAndMatchesGenerator()
    {
        Assert.SkipUnless(File.Exists(Path), $"model not found: {Path}");
        var cpu = new CpuExecutionProvider();
        using var device = cpu.OpenDevice(cpu.GetDevices()[0]);
        using var model = LlmModel.Load(Path, new LlmLoadOptions { Devices = [new DeviceSlot { Device = device }], ContextLength = 64, MaxBatchTokens = 64 });
        var options = new GenerationOptions { MaxNewTokens = 16, StopAtEog = false, Diffusion = new DiffusionOptions { Steps = 8 } };
        const string prompt = "The quick brown fox";

        string expected;
        using (var generator = model.CreateDiffusionGenerator())
        {
            expected = model.Tokenizer.Decode(generator.Generate(model.Tokenizer.Encode(prompt, addSpecial: false, parseSpecial: true), options));
        }

        using var client = new SharplightChatClient(model);
        var backend = Assert.IsType<DiffusionGenerationBackend>(client.Backend);
        var result = new TextGenerationResult();
        var text = new System.Text.StringBuilder();
        await foreach (var piece in backend.GenerateAsync(new TextGenerationRequest { Prompt = prompt, Options = options }, result, TestContext.Current.CancellationToken))
        {
            text.Append(piece);
        }

        Assert.Equal(expected, text.ToString());
        Assert.Equal(16, result.GeneratedTokens);
        Assert.True(result.ReachedTokenLimit);
        Assert.Equal(64, backend.ContextLength);
    }
}
