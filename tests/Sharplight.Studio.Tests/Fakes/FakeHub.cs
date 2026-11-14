using Sharplight.Hub;
using Sharplight.Studio.Services.Hub;

namespace Sharplight.Studio.Tests.Fakes;

/// <summary>A hub whose pulls report progress and then wait for <see cref="Release"/> (or cancellation).</summary>
internal sealed class FakeHub : IHubService
{
    private TaskCompletionSource _gate = new(TaskCreationOptions.RunContinuationsAsynchronously);

    public List<ModelReference> Discarded { get; } = [];

    public List<HubModelSummary> Results { get; } = [];

    /// <summary>The progress a pull reports before it waits.</summary>
    public PullUpdate FirstUpdate { get; set; } = new(PullPhase.Downloading, "Pulling", Received: 400, Total: 1000, File: "model.gguf", FileReceived: 400, FileSize: 1000);

    private int _pulls;

    public int Pulls => _pulls;

    /// <summary>Signalled when a pull reported its first progress.</summary>
    public TaskCompletionSource Started { get; private set; } = new(TaskCreationOptions.RunContinuationsAsynchronously);

    public void Release() => _gate.TrySetResult();

    public void Reset()
    {
        _gate = new(TaskCreationOptions.RunContinuationsAsynchronously);
        Started = new(TaskCreationOptions.RunContinuationsAsynchronously);
    }

    public Task<IReadOnlyList<HubModelSummary>> SearchAsync(HubSearchQuery query, CancellationToken cancellationToken) =>
        Task.FromResult<IReadOnlyList<HubModelSummary>>([.. Results]);

    public Task<RepositoryDetails> GetDetailsAsync(string repository, CancellationToken cancellationToken) =>
        Task.FromResult(new RepositoryDetails(new HubRepository(repository, "main", "abc", false, [], "qwen3"), ModelFormat.Gguf,
        [
            new VariantInfo("Q4_K_M", 5_030_000_000, 1, false, true),
            new VariantInfo("Q5_K_M", 5_850_000_000, 1, false, false),
            new VariantInfo("Q8_0", 8_710_000_000, 1, false, false),
        ]));

    public async Task<ModelManifest> PullAsync(ModelReference reference, IProgress<PullUpdate> progress, CancellationToken cancellationToken)
    {
        Interlocked.Increment(ref _pulls);
        progress.Report(FirstUpdate);
        Started.TrySetResult();
        await _gate.Task.WaitAsync(cancellationToken);
        progress.Report(new PullUpdate(PullPhase.Converting, "Converting"));
        return new ModelManifest { Repository = reference.Repository, Tag = reference.Tag ?? "", Alias = "test", Commit = "abc" };
    }

    public void DiscardPartial(ModelReference reference) => Discarded.Add(reference);
}
