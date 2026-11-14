using System.Runtime.CompilerServices;
using System.Threading.Channels;
using Sharplight.Llm.Serving;
using Sharplight.Llm.Serving.Hosting;
using Sharplight.Studio.Models;

namespace Sharplight.Studio.Services.Runtime;

/// <summary>
/// A model registered by <see cref="EngineModelRuntime"/> on its <see cref="ModelHost"/>. Each request takes a lease (loading the
/// model when it sleeps). Text conversations run through the model's batching engine; conversations with images run through a
/// single-sequence session with the model's vision encoder (<see cref="VisionChat"/>), because the batching engine has no image
/// input yet.
/// </summary>
internal sealed class EngineLoadedModel : ILoadedModel, IDisposable
{
    private const int CharactersPerToken = 4;

    private readonly ModelHost _host;
    private readonly Action<EngineLoadedModel, MessageStats> _visionFinished;
    private readonly Action _changed;
    private readonly Lock _lock = new();
    private bool _reasoningTemplate;
    private Func<string, int>? _countTokens;
    private ChatModelInstance? _visionOwner;
    private VisionChat? _vision;
    private LoadProgressTracker _tracker = new();
    private bool _hasProgress;

    public EngineLoadedModel(ModelLoadRequest request, ComputeDeviceOption device, ModelHost host, bool pinned, ModelCapabilities capabilities,
        Action<EngineLoadedModel, MessageStats> visionFinished, Action changed)
    {
        Id = request.Id;
        Path = request.Path;
        Device = device;
        Settings = request.Settings;
        IsPinned = pinned;
        Capabilities = capabilities | ModelCapabilities.Chat;
        ContextLength = request.Settings.ContextLength;
        _host = host;
        _visionFinished = visionFinished;
        _changed = changed;
        LoadedAt = DateTimeOffset.UtcNow;
    }

    /// <inheritdoc/>
    public string Id { get; }

    /// <inheritdoc/>
    public string Path { get; }

    /// <inheritdoc/>
    public ComputeDeviceOption Device { get; }

    /// <inheritdoc/>
    public LoadSettings Settings { get; }

    /// <inheritdoc/>
    public ModelCapabilities Capabilities { get; private set; }

    /// <inheritdoc/>
    public int ContextLength { get; private set; }

    /// <inheritdoc/>
    public bool IsPinned { get; }

    /// <inheritdoc/>
    public DateTimeOffset LoadedAt { get; }

    /// <inheritdoc/>
    public LoadStatus? LoadStatus { get; private set; }

    /// <summary>Receives load progress while <see cref="EngineModelRuntime.LoadAsync"/> waits for this model.</summary>
    public IProgress<LoadStatus>? LoadListener { get; set; }

    /// <summary>Cancelled to abandon a load in progress (the model is being removed).</summary>
    public CancellationTokenSource Abandon { get; } = new();

    /// <inheritdoc/>
    public ResidentState State => ToState(_host.Get(Id));

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

    /// <summary>Maps the host's view of a model to the app's states.</summary>
    internal static ResidentState ToState(ModelInfo? info) => info?.State switch
    {
        ModelState.Loading => ResidentState.Loading,
        ModelState.Loaded => info.ActiveLeases > 0 ? ResidentState.Busy : ResidentState.Ready,
        ModelState.Unloading => ResidentState.Unloading,
        ModelState.Failed => ResidentState.Failed,
        _ => ResidentState.Sleeping,
    };

    /// <inheritdoc/>
    public IAsyncEnumerable<ChatStreamUpdate> StreamChatAsync(ChatInput input, CancellationToken cancellationToken)
    {
        if (input.HasImages && _countTokens is not null && !Capabilities.HasFlag(ModelCapabilities.Vision))
        {
            throw new NotSupportedException($"{Id} has no vision tower the engine can run, so it cannot read images.");
        }

        return StreamAsync(input, cancellationToken);
    }

    /// <inheritdoc/>
    public int CountTokens(string text) => _countTokens?.Invoke(text) ?? (text.Length + CharactersPerToken - 1) / CharactersPerToken;

    /// <inheritdoc/>
    public EngineMetricsSnapshot? GetMetrics()
    {
        foreach (var (name, instance) in _host.LoadedInstances())
        {
            if (name == Id && instance is ChatModelInstance chat)
            {
                return chat.Engine.GetMetrics();
            }
        }

        return null;
    }

    /// <summary>Called by the host's loader for every new instance: learns the capabilities and the tokenizer.</summary>
    internal void OnInstanceLoaded(LoadedModel loaded)
    {
        if (loaded is not ChatModelInstance instance)
        {
            return;
        }

        var template = instance.Model?.ChatTemplate;
        _reasoningTemplate = template?.Contains("<think>", StringComparison.Ordinal) == true
            || template?.Contains("enable_thinking", StringComparison.Ordinal) == true;
        var capabilities = TemplateCapabilities.From(template) | ModelCapabilities.Chat;
        if (instance.Model is { SupportsImages: true })
        {
            capabilities |= ModelCapabilities.Vision;
        }

        if (instance.Model is { HasDraftHead: true })
        {
            capabilities |= ModelCapabilities.DraftHead;
        }

        Capabilities = capabilities;
        ContextLength = instance.Engine.ContextLength;
        var tokenizer = instance.Service.Tokenizer;
        _countTokens = text => tokenizer.Encode(text, addSpecial: false, parseSpecial: false).Length;
        LoadStatus = null;
        _tracker = new LoadProgressTracker();
        _hasProgress = false;
        _changed();
    }

    /// <summary>A load log line: shown until structured progress arrives.</summary>
    internal void OnLoadMessage(string message)
    {
        if (_hasProgress)
        {
            return;
        }

        Publish(Runtime.LoadStatus.Message(message));
    }

    /// <summary>A structured progress report of the engine.</summary>
    internal void OnLoadProgress(Sharplight.Runtime.LoadProgress progress)
    {
        _hasProgress = true;
        Publish(_tracker.Report(progress));
    }

    private void Publish(LoadStatus status)
    {
        LoadStatus = status;
        LoadListener?.Report(status);
        _changed();
    }

    /// <summary>Frees the vision session when its instance is gone.</summary>
    internal void OnHostChanged()
    {
        if (State is ResidentState.Ready or ResidentState.Busy or ResidentState.Loading)
        {
            return;
        }

        LoadStatus = null;
        _tracker = new LoadProgressTracker();
        _hasProgress = false;
        ReleaseVision();
    }

    /// <summary>Abandons a load in progress and disposes the vision session.</summary>
    public void Dispose()
    {
        Abandon.Cancel();
        ReleaseVision();
    }

    private void ReleaseVision()
    {
        lock (_lock)
        {
            _vision?.Dispose();
            _vision = null;
            _visionOwner = null;
        }
    }

    private async IAsyncEnumerable<ChatStreamUpdate> StreamAsync(ChatInput input, [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        ModelLease lease;
        try
        {
            lease = await _host.AcquireAsync(Id, ModelKind.Chat, cancellationToken).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            lease = null!;
        }

        if (lease is null)
        {
            yield return ChatStreamUpdate.Finished(new MessageStats { Finish = "cancelled" });
            yield break;
        }

        using (lease)
        {
            var instance = lease.As<ChatModelInstance>();
            var updates = input.HasImages ? StreamVisionAsync(instance, input, cancellationToken) : StreamTextAsync(instance, input, cancellationToken);
            await foreach (var update in updates.ConfigureAwait(false))
            {
                yield return update;
            }
        }
    }

    private async IAsyncEnumerable<ChatStreamUpdate> StreamTextAsync(ChatModelInstance instance, ChatInput input, [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var request = ChatRequestMapper.ToChatRequest(input, _reasoningTemplate);
        var run = instance.Service.StartChat(request, cancellationToken);
        await foreach (var chatEvent in run.ReadAllAsync(CancellationToken.None).ConfigureAwait(false))
        {
            if (chatEvent.Content is { Length: > 0 } content)
            {
                yield return ChatStreamUpdate.Content(content);
            }
            else if (chatEvent.Reasoning is { Length: > 0 } reasoning)
            {
                yield return ChatStreamUpdate.Reasoning(reasoning);
            }
            else if (chatEvent.ToolCall is { } call)
            {
                yield return new ChatStreamUpdate(ChatStreamKind.ToolCall, ChatRequestMapper.FormatToolCall(call.Name, call.ArgumentsJson));
            }
            else if (chatEvent.Finish is { } finish)
            {
                var stats = ChatRequestMapper.ToStats(chatEvent.Usage ?? default, chatEvent.Timings, finish);
                yield return ChatStreamUpdate.Finished(stats, chatEvent.Error);
            }
        }
    }

    private VisionChat GetVision(ChatModelInstance instance)
    {
        lock (_lock)
        {
            if (_visionOwner != instance)
            {
                _vision?.Dispose();
                _vision = instance.Model is { SupportsImages: true } model
                    ? new VisionChat(model)
                    : throw new NotSupportedException($"{Id} has no vision tower the engine can run, so it cannot read images.");
                _visionOwner = instance;
            }

            return _vision!;
        }
    }

    private async IAsyncEnumerable<ChatStreamUpdate> StreamVisionAsync(ChatModelInstance instance, ChatInput input, [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var vision = GetVision(instance);

        // The session runs on its own thread; a channel hands its events to the caller.
        var channel = Channel.CreateUnbounded<ChatStreamUpdate>(new UnboundedChannelOptions { SingleReader = true, SingleWriter = true });
        var producer = Task.Run(async () =>
        {
            try
            {
                var stats = await vision.RunAsync(input, _reasoningTemplate, update => channel.Writer.TryWrite(update), cancellationToken).ConfigureAwait(false);
                channel.Writer.TryWrite(ChatStreamUpdate.Finished(stats));
                _visionFinished(this, stats);
            }
            catch (OperationCanceledException)
            {
                channel.Writer.TryWrite(ChatStreamUpdate.Finished(new MessageStats { Finish = "cancelled" }));
            }
            catch (Exception error)
            {
                channel.Writer.TryWrite(ChatStreamUpdate.Finished(new MessageStats { Finish = "error" }, error.Message));
            }
            finally
            {
                channel.Writer.TryComplete();
            }
        }, CancellationToken.None);

        await foreach (var update in channel.Reader.ReadAllAsync(CancellationToken.None).ConfigureAwait(false))
        {
            yield return update;
        }

        await producer.ConfigureAwait(false);
    }
}
