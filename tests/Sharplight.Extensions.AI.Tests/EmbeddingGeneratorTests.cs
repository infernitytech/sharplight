using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Sharplight.Cpu;
using Sharplight.Gguf;
using Sharplight.Llm.Embeddings;
using Sharplight.Runtime;

namespace Sharplight.Extensions.AI.Tests;

public class EmbeddingGeneratorTests
{
    sealed class Fake : ITextEmbedder, IDisposable
    {
        public string Name => "fake";
        public int Dimensions => 4;
        public int MaxSequenceLength => 100;
        public IReadOnlyDictionary<string, string> Prompts { get; } = new Dictionary<string, string> { ["query"] = "q: " };
        public EmbedOptions? Last;
        public bool Disposed;

        public EmbeddingResult Embed(IReadOnlyList<string> texts, EmbedOptions? options = null)
        {
            Last = options;
            return new([.. texts.Select(t => new float[] { t.Length, 1, 2, 3 }.Take(options?.Dimensions ?? 4).ToArray())], [.. texts.Select(t => t.Length)]);
        }

        public EmbeddingResult EmbedTokens(IReadOnlyList<IReadOnlyList<int>> tokens, EmbedOptions? options = null) => throw new NotSupportedException();
        public void Dispose() => Disposed = true;
    }

    [Fact]
    public async Task GeneratesInOrderWithUsageAndOptions()
    {
        var fake = new Fake();
        using var gen = new SharplightEmbeddingGenerator(fake);
        var r = await gen.GenerateAsync(["ab", "abcd"], new EmbeddingGenerationOptions { Dimensions = 2, AdditionalProperties = new() { ["input_type"] = "query" } });
        Assert.Equal(2, r.Count);
        Assert.Equal([2f, 1f], r[0].Vector.ToArray());
        Assert.Equal(4f, r[1].Vector.Span[0]);
        Assert.Equal("fake", r[0].ModelId);
        Assert.Equal(6, r.Usage!.InputTokenCount);
        Assert.Equal(2, fake.Last!.Dimensions);
        Assert.Equal("query", fake.Last.Prompt);

        await gen.GenerateAsync(["x"], new EmbeddingGenerationOptions { AdditionalProperties = new() { ["input_type"] = "document" } });
        Assert.Null(fake.Last!.Prompt);
        var md = gen.GetService<EmbeddingGeneratorMetadata>()!;
        Assert.Equal("sharplight", md.ProviderName);
        Assert.Equal(4, md.DefaultModelDimensions);
        Assert.Same(fake, gen.GetService<ITextEmbedder>());
        Assert.False(fake.Disposed);   // not owned
    }

    [Fact]
    public async Task DependencyInjectionRegistration()
    {
        var fake = new Fake();
        var services = new ServiceCollection();
        services.AddSharplightEmbeddingGenerator(fake);
        await using var sp = services.BuildServiceProvider();
        var gen = sp.GetRequiredService<IEmbeddingGenerator<string, Embedding<float>>>();
        var v = await gen.GenerateVectorAsync("hello");
        Assert.Equal(5f, v.Span[0]);
    }

    [Fact]
    public async Task RealModelOnCpu()
    {
        string path = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models/gguf/all-MiniLM-L6-v2/all-MiniLM-L6-v2-ggml-model-f16.gguf");
        Assert.SkipUnless(File.Exists(path), $"Model not found: {path}");
        GgufImporter.Register();
        var cpu = new CpuExecutionProvider();
        var services = new ServiceCollection();
        services.AddSharplightEmbeddingGenerator(path, new EmbeddingLoadOptions { Devices = [new DeviceSlot { Device = cpu.OpenDevice(cpu.GetDevices()[0]) }] });
        await using var sp = services.BuildServiceProvider();
        var gen = sp.GetRequiredService<IEmbeddingGenerator<string, Embedding<float>>>();
        var r = await gen.GenerateAsync(["The cat sits on the mat.", "A cat is sitting on a mat.", "Stock markets fell sharply today."]);
        Assert.Equal(384, r[0].Vector.Length);
        float Dot(int a, int b) { float s = 0; for (int i = 0; i < 384; i++) s += r[a].Vector.Span[i] * r[b].Vector.Span[i]; return s; }
        Assert.Equal(1f, Dot(0, 0), 3);
        Assert.True(Dot(0, 1) > Dot(0, 2) + 0.3f, $"similar {Dot(0, 1)} vs unrelated {Dot(0, 2)}");
    }
}
