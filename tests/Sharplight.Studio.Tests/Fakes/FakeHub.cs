using Sharplight.Hub;
using Sharplight.Studio.Services.Compatibility;
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

    /// <summary>The dimensions <see cref="ProbeAsync"/> reports (an 8B model with grouped-query attention by default).</summary>
    public ModelShape? Shape { get; set; } = new(36, 32, 8, 4096, 128, 128, 40960, 151936);

    /// <summary>
    /// The verdict <see cref="ProbeAsync"/> reports for a variant whose weight formats are read (repository, variant tag); by default
    /// a supported GGUF architecture.
    /// </summary>
    public Func<string, string, ModelCompatibility> Compatibility { get; set; } =
        (_, _) => new ModelCompatibility(CompatibilityState.Supported, "qwen3", "The GGUF architecture 'qwen3' is supported. Tensor types: Q4_K (145), F32 (113), Q6_K (36), all with kernels.");

    /// <summary>Variants that were probed with their weight formats, in order.</summary>
    public List<string> ProbedVariants { get; } = [];

    /// <summary>The variants <see cref="GetDetailsAsync"/> lists, by repository; three Qwen3-8B quantizations by default.</summary>
    public Func<string, IReadOnlyList<VariantInfo>> Variants { get; set; } = _ =>
    [
        new VariantInfo("Q4_K_M", 5_030_000_000, 1, false, true),
        new VariantInfo("Q5_K_M", 5_850_000_000, 1, false, false),
        new VariantInfo("Q8_0", 8_710_000_000, 1, false, false),
    ];

    public Task<ModelProbe> ProbeAsync(RepositoryDetails details, VariantInfo variant, bool readWeightFormats, CancellationToken cancellationToken)
    {
        var verdict = Compatibility(details.Repository.Id, variant.Tag);
        lock (ProbedVariants)
        {
            ProbedVariants.Add(variant.Tag);
        }

        var architecture = verdict.State == CompatibilityState.UnsupportedWeightFormat
            ? new ModelCompatibility(CompatibilityState.Supported, details.Repository.Architecture, "The architecture is supported.", WeightsChecked: false)
            : verdict with { WeightsChecked = false };
        return Task.FromResult(new ModelProbe(Shape, readWeightFormats ? verdict : architecture, architecture));
    }

    public Task<RepositoryDetails> GetDetailsAsync(string repository, CancellationToken cancellationToken) =>
        Task.FromResult(new RepositoryDetails(new HubRepository(repository, "main", "abc", false, [], "qwen3"), ModelFormat.Gguf, Variants(repository)));

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
