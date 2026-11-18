using System.Runtime.CompilerServices;
using Sharplight.Llm.Serving;
using Sharplight.Speech;
using Sharplight.Tasks;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Runtime;

namespace Sharplight.Studio.Tests.Fakes;

/// <summary>A resident model that streams a scripted answer.</summary>
internal sealed class FakeModel(string id, ModelCapabilities capabilities = ModelCapabilities.Chat) : ILoadedModel
{
    public string Id { get; } = id;

    public string Kind { get; init; } = Sharplight.Models.ModelKinds.TextGeneration;

    public string Path { get; init; } = $"/models/{id}.slm";

    public ComputeDeviceOption Device { get; init; } = FakeRuntime.Gpu;

    public LoadSettings Settings { get; init; } = new();

    public IReadOnlyList<(string Role, string Path)> Auxiliary { get; init; } = [];

    public ModelCapabilities Capabilities { get; } = capabilities;

    public int ContextLength => Settings.ContextLength;

    public ResidentState State { get; set; } = ResidentState.Ready;

    public string? Error => null;

    public LoadStatus? LoadStatus { get; set; }

    public int ActiveRequests { get; set; }

    public long ResidentBytes { get; set; } = 1L << 30;

    public long EstimatedBytes { get; set; } = 1L << 30;

    public bool IsPinned { get; set; } = true;

    public DateTimeOffset LoadedAt { get; } = DateTimeOffset.UtcNow;

    public DateTimeOffset? LastUsed { get; set; }

    /// <summary>The updates every request streams.</summary>
    public List<ChatStreamUpdate> Script { get; } =
    [
        ChatStreamUpdate.Reasoning("Thinking about it."),
        ChatStreamUpdate.Content("Hello"),
        ChatStreamUpdate.Content(" there!"),
        ChatStreamUpdate.Finished(new MessageStats { PromptTokens = 12, CompletionTokens = 3, PrefillMs = 10, DecodeMs = 20 }),
    ];

    /// <summary>When set, the stream waits on it after the first update (until cancelled).</summary>
    public TaskCompletionSource? Gate { get; set; }

    /// <summary>Blocking work per step (a stand-in for a graph run that holds its thread), or zero.</summary>
    public TimeSpan StepWork { get; set; }

    /// <summary>Inputs received.</summary>
    public List<ChatInput> Inputs { get; } = [];

    public async IAsyncEnumerable<ChatStreamUpdate> StreamChatAsync(ChatInput input, [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        Inputs.Add(input);
        for (int i = 0; i < Script.Count; i++)
        {
            await Task.Yield();
            if (i == 1 && Gate is { } gate)
            {
                await gate.Task.WaitAsync(cancellationToken);
            }

            cancellationToken.ThrowIfCancellationRequested();
            SlowWork.Block(StepWork);
            yield return Script[i];
        }
    }

    /// <summary>The task clients of the model, by interface type (<see cref="ISpeechRecognizer"/>, <see cref="ISpeechSynthesizer"/>).</summary>
    public Dictionary<Type, ITaskClient> Clients { get; } = [];

    /// <summary>Leases taken so far.</summary>
    public int Leases { get; private set; }

    public Task<TaskClientLease<TClient>> AcquireClientAsync<TClient>(CancellationToken cancellationToken)
        where TClient : class, ITaskClient
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (!Clients.TryGetValue(typeof(TClient), out var client))
        {
            throw new NotSupportedException($"{Id} has no {typeof(TClient).Name}.");
        }

        Leases++;
        return Task.FromResult(new TaskClientLease<TClient>((TClient)client));
    }

    /// <summary>The embedder of an embedding model (kind <c>embeddings</c>).</summary>
    public Sharplight.Llm.Embeddings.ITextEmbedder? Embedder { get; set; }

    public Task<EmbedderLease> AcquireEmbedderAsync(CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        Leases++;
        return Task.FromResult(new EmbedderLease(Embedder ?? throw new NotSupportedException($"{Id} is not an embedding model.")));
    }

    /// <summary>The reranker of a reranking model (kind <c>reranker</c>).</summary>
    public Sharplight.Llm.Embeddings.ITextReranker? Reranker { get; set; }

    public Task<RerankerLease> AcquireRerankerAsync(CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        Leases++;
        return Task.FromResult(new RerankerLease(Reranker ?? throw new NotSupportedException($"{Id} is not a reranking model.")));
    }

    public int CountTokens(string text) => (text.Length + 3) / 4;

    public EngineMetricsSnapshot? GetMetrics() => null;
}
