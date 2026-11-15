using System.Text;
using Microsoft.Extensions.DependencyInjection;
using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Runtime;

namespace Sharplight.Serving.Tests;

/// <summary>
/// The text-generation task client on a real small model (skipped when the file is absent): the same greedy completion from the
/// classic <see cref="Generator"/>, from <see cref="LlmModel.CreateTextGenerator"/> and from an <see cref="ITextGenerator"/> resolved
/// through the service container (task model load path).
/// </summary>
[Collection("RealModel")]
public class RealModelTextGeneratorTests
{
    private const string Prompt = "The capital of France is";
    private static readonly string ModelPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models/gguf/Qwen3-0.6B/Qwen3-0.6B-Q8_0.gguf");
    private static readonly GenerationOptions Greedy = new() { MaxNewTokens = 12, Sampling = SamplingOptions.Greedy };

    [Fact]
    public async Task TextGeneratorMatchesTheGeneratorThroughEveryEntryPoint()
    {
        Assert.SkipUnless(File.Exists(ModelPath), "model not found: " + ModelPath);
        var cancellation = TestContext.Current.CancellationToken;
        var available = TestHost.Providers.Discover().Where(x => x.Availability.IsAvailable).ToList();
        var provider = available.Where(x => x.Provider.Name != "cpu").Concat(available).First().Provider;
        using var device = provider.OpenDevice(provider.GetDevices()[0], new DeviceOptions());
        var slot = new DeviceSlot { Device = device };

        string expected;
        using (var model = LlmModel.Load(TestHost.Loader, ModelPath, new LlmLoadOptions { Devices = [slot], ContextLength = 1024 }))
        {
            Assert.Equal(ModelKinds.TextGeneration, model.Description.Kind);
            using (var session = model.CreateSession())
            {
                expected = string.Concat(new Generator(session, model.Tokenizer).Generate(Prompt, Greedy).Select(t => t.Text));
            }

            using var generator = model.CreateTextGenerator();
            var first = await generator.GenerateAsync(Prompt, Greedy, cancellation);
            Assert.Equal(expected, first.Text);
            Assert.Equal(12, first.Stats!.GeneratedTokens);
            var second = new StringBuilder();
            await foreach (var token in generator.GenerateStreamingAsync(Prompt, Greedy, cancellation))
            {
                second.Append(token.Text);
            }

            Assert.Equal(expected, second.ToString());
            Assert.Same(model, generator.GetService(typeof(LlmModel)));
        }

        var services = new ServiceCollection().AddTextGenerator(o =>
        {
            o.ModelPath = ModelPath;
            o.Devices = [slot];
            o.ContextLength = 1024;
        });
        await using var container = services.BuildServiceProvider();
        var resolved = container.GetRequiredService<ITextGenerator>();
        Assert.Equal(ModelKinds.TextGeneration, resolved.Metadata.Kind);
        Assert.Equal(expected, (await resolved.GenerateAsync(Prompt, Greedy, cancellation)).Text);

        using var cancelled = new CancellationTokenSource();
        await cancelled.CancelAsync();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => resolved.GenerateAsync(Prompt, Greedy, cancelled.Token));
    }
}
