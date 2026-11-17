using Sharplight.Hosting.Loaders;
using Sharplight.Llm.Serving;
using Sharplight.Llm.Serving.Hosting;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Threading;
using Sharplight.Tasks;

namespace Sharplight.Studio.Services.Runtime;

/// <summary>
/// A task-layer model (speech recognition, speech synthesis, ...) registered by <see cref="EngineModelRuntime"/> on its
/// <see cref="ModelHost"/>: it is loaded, evicted, unloaded for idleness and reloaded on demand like a language model. Runs lease the
/// model's task client (<see cref="AcquireClientAsync{TClient}"/>).
/// </summary>
internal sealed class EngineTaskModel : IEngineModel
{
    private readonly ModelHost _host;
    private readonly Action _changed;
    private readonly UpdateThrottle _publish;
    private LoadProgressTracker _tracker = new();
    private bool _hasProgress;

    public EngineTaskModel(ModelLoadRequest request, ComputeDeviceOption device, ModelHost host, bool pinned, Action changed)
    {
        Id = request.Id;
        Kind = request.Kind;
        Path = request.Path;
        Device = device;
        Settings = request.Settings;
        Auxiliary = request.Auxiliary ?? [];
        IsPinned = pinned;
        Capabilities = request.Capabilities;
        _host = host;
        _changed = changed;
        _publish = new UpdateThrottle(PublishNow);
        LoadedAt = DateTimeOffset.UtcNow;
    }

    /// <inheritdoc/>
    public string Id { get; }

    /// <inheritdoc/>
    public string Kind { get; }

    /// <inheritdoc/>
    public string Path { get; }

    /// <inheritdoc/>
    public ComputeDeviceOption Device { get; }

    /// <inheritdoc/>
    public LoadSettings Settings { get; }

    /// <inheritdoc/>
    public IReadOnlyList<(string Role, string Path)> Auxiliary { get; }

    /// <inheritdoc/>
    public ModelCapabilities Capabilities { get; }

    /// <inheritdoc/>
    public int ContextLength => Settings.ContextLength;

    /// <inheritdoc/>
    public bool IsPinned { get; }

    /// <inheritdoc/>
    public DateTimeOffset LoadedAt { get; }

    /// <inheritdoc/>
    public LoadStatus? LoadStatus { get; private set; }

    /// <inheritdoc/>
    public IProgress<LoadStatus>? LoadListener { get; set; }

    /// <inheritdoc/>
    public CancellationTokenSource Abandon { get; } = new();

    /// <inheritdoc/>
    public ResidentState State => EngineLoadedModel.ToState(_host.Get(Id));

    /// <inheritdoc/>
    public string? Error => _host.Get(Id)?.Error;

    /// <inheritdoc/>
    public int ActiveRequests => _host.Get(Id)?.ActiveLeases ?? 0;

    /// <inheritdoc/>
    public long ResidentBytes => _host.Get(Id)?.ResidentBytes ?? 0;

    /// <inheritdoc/>
    public long EstimatedBytes => _host.Get(Id)?.EstimatedBytes ?? 0;

    /// <inheritdoc/>
    public DateTimeOffset? LastUsed => _host.Get(Id)?.LastUsed;

    /// <inheritdoc/>
    public IAsyncEnumerable<ChatStreamUpdate> StreamChatAsync(ChatInput input, CancellationToken cancellationToken)
        => throw new NotSupportedException($"{Id} is a {Kind} model, not a chat model.");

    /// <inheritdoc/>
    public async Task<TaskClientLease<TClient>> AcquireClientAsync<TClient>(CancellationToken cancellationToken)
        where TClient : class, ITaskClient
    {
        var lease = await _host.AcquireAsync(Id, ModelKind.Task, cancellationToken).ConfigureAwait(false);
        try
        {
            return new TaskClientLease<TClient>(lease.As<TaskModelInstance>().GetClient<TClient>(), lease);
        }
        catch
        {
            lease.Dispose();
            throw;
        }
    }

    /// <inheritdoc/>
    public int CountTokens(string text) => 0;

    /// <inheritdoc/>
    public EngineMetricsSnapshot? GetMetrics() => null;

    /// <summary>Called by the host's loader for every new instance.</summary>
    internal void OnInstanceLoaded(LoadedModel _)
    {
        LoadStatus = null;
        _tracker = new LoadProgressTracker();
        _hasProgress = false;
        _changed();
    }

    /// <summary>A load log line: shown until structured progress arrives (ignored once the model is loaded).</summary>
    internal void OnLoadMessage(string message)
    {
        if (State == ResidentState.Loading && !_hasProgress)
        {
            Publish(Runtime.LoadStatus.Message(message));
        }
    }

    /// <summary>A structured progress report of the engine.</summary>
    internal void OnLoadProgress(Sharplight.Runtime.LoadProgress progress)
    {
        if (State != ResidentState.Loading)
        {
            // A speech client compiles more kernels on its first request; that is not a model load.
            return;
        }

        _hasProgress = true;
        Publish(_tracker.Report(progress));
    }

    /// <inheritdoc/>
    public void OnHostChanged()
    {
        if (State is ResidentState.Ready or ResidentState.Busy or ResidentState.Loading)
        {
            return;
        }

        LoadStatus = null;
        _tracker = new LoadProgressTracker();
        _hasProgress = false;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        Abandon.Cancel();
        _publish.Dispose();
    }

    /// <summary>
    /// Shows a load status: the engine reports per kernel and per uploaded tensor, so listeners and <see cref="IModelRuntime.ModelsChanged"/>
    /// hear about it at most about 15 times a second (the latest status), and at once when the load is done.
    /// </summary>
    private void Publish(LoadStatus status)
    {
        LoadStatus = status;
        _publish.Request();
        if (status.IsDone)
        {
            _publish.Flush();
        }
    }

    private void PublishNow()
    {
        if (LoadStatus is { } status)
        {
            LoadListener?.Report(status);
        }

        _changed();
    }
}
