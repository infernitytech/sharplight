using Microsoft.Extensions.DependencyInjection;
using Sharplight.Hosting.Loaders;
using Sharplight.Llm.Embeddings;
using Sharplight.Llm.Serving;
using Sharplight.Llm.Serving.Chat;
using Sharplight.Llm.Serving.Hosting;

namespace Sharplight.Hosting.Tests;

/// <summary>
/// Real small models on the CPU through <c>AddSharplight()</c> + <c>AddSharplightHosting()</c> (skipped when the files are missing): a
/// chat model, an embedder and a reranker load from options into one host and answer.
/// </summary>
public sealed class RealModelHostingTests
{
    private static readonly string Home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
    private static readonly string Chat = Path.Combine(Home, "models/gguf/SmolLM2-135M/SmolLM2-135M.Q8_0.gguf");
    private static readonly string Embedder = Path.Combine(Home, "models/gguf/all-MiniLM-L6-v2/all-MiniLM-L6-v2.F16.gguf");
    private static readonly string Reranker = Path.Combine(Home, "models/gguf/bge-reranker-v2-m3/bge-reranker-v2-m3-Q8_0.gguf");

    [Fact]
    public async Task ChatEmbeddingAndRerankerLoadOnTheCpu()
    {
        Assert.SkipUnless(File.Exists(Chat) && File.Exists(Embedder) && File.Exists(Reranker), "models not found");
        var services = new ServiceCollection();
        services.AddSharplight();
        services.AddSharplightHosting(options =>
        {
            options.Reserve = 0;
            options.Defaults.Device = DeviceSpecifications.Cpu;
            options.Defaults.ContextLength = 512;
            options.Defaults.BatchTokens = 64;
            options.Defaults.Slots = 1;
            options.Models.Add(new HostedModelOptions { Name = "chat", Path = Chat });
            options.Models.Add(new HostedModelOptions { Name = "embed", Path = Embedder, Kind = ModelKind.Embedding, Lazy = true });
            options.Models.Add(new HostedModelOptions { Name = "rerank", Path = Reranker, Kind = ModelKind.Reranker, Lazy = true });
        });
        await using var provider = services.BuildServiceProvider();
        var manager = provider.GetRequiredService<IModelHostManager>();
        await manager.Host.LoadStartupModelsAsync(TestContext.Current.CancellationToken);

        using (var lease = await manager.Host.AcquireAsync("chat", ModelKind.Chat, TestContext.Current.CancellationToken))
        {
            var service = lease.As<LoadedModel>() is IChatModel chat ? chat.Service : throw new InvalidOperationException();
            var request = new CompletionRequest { Prompt = "The capital of France is", MaxTokens = 4, Sampling = new SamplingParams { Temperature = 0 } };
            var result = await service.StartCompletion(request, TestContext.Current.CancellationToken).CompleteAsync(TestContext.Current.CancellationToken);
            Assert.True(result.Usage.CompletionTokens > 0);
        }

        using (var lease = await manager.Host.AcquireAsync("embed", ModelKind.Embedding, TestContext.Current.CancellationToken))
        {
            var embedder = ((ITaskClientModel)lease.As<LoadedModel>()).FindClient<ITextEmbedder>()!;
            Assert.Equal(384, (await embedder.EmbedAsync(["hello"], cancellationToken: TestContext.Current.CancellationToken)).Vectors[0].Length);
        }

        using (var lease = await manager.Host.AcquireAsync("rerank", ModelKind.Reranker, TestContext.Current.CancellationToken))
        {
            var reranker = ((ITaskClientModel)lease.As<LoadedModel>()).FindClient<ITextReranker>()!;
            var scores = await reranker.RerankAsync("capital of France", ["Paris is the capital of France.", "Bananas are yellow."], cancellationToken: TestContext.Current.CancellationToken);
            Assert.True(scores.Scores[0] > scores.Scores[1]);
        }

        var status = manager.GetStatus();
        Assert.Equal(3, status.LoadedModels);
        Assert.All(status.Models, model => Assert.Equal(["cpu:0"], model.Devices));
        Assert.Contains(status.Devices, device => device.Id == "cpu:0" && device.IsOpen && device.Models.Count == 3);
    }
}
