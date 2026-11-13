using Sharplight.Llm.Serving.Chat;

namespace Sharplight.Llm.Serving.Hosting;

/// <summary>A loaded chat model: weights on the devices, a batch decoder, its own scheduler thread (<see cref="Engine"/>) and chat layer.</summary>
public sealed class ChatModelInstance : LoadedModel, IChatModel
{
    private readonly LlmModel? _model;
    private readonly bool _ownsEngine;
    /// <inheritdoc/>
    public InferenceEngine Engine { get; }
    /// <inheritdoc/>
    public ModelService Service { get; }
    private readonly long _resident;

    internal ChatModelInstance(LlmModel? model, InferenceEngine engine, ModelService service, long resident, bool ownsEngine = true)
    {
        _model = model;
        Engine = engine;
        Service = service;
        _resident = resident;
        _ownsEngine = ownsEngine;
    }

    /// <summary>Wraps a service built elsewhere (the caller keeps ownership of its engine and model).</summary>
    public static ChatModelInstance Wrap(ModelService service, long residentBytes = 0) => new(null, service.Engine, service, residentBytes, ownsEngine: false);

    /// <inheritdoc/>
    public override ModelKind Kind => ModelKind.Chat;
    /// <inheritdoc/>
    public override long ResidentBytes => _resident;
    /// <inheritdoc/>
    public override bool IsIdle
    {
        get
        {
            var metrics = Engine.GetMetrics();
            return metrics.ActiveSequences == 0 && metrics.QueueDepth == 0;
        }
    }

    /// <inheritdoc/>
    public override Task<bool> DrainAsync(TimeSpan timeout) => Engine.DrainAsync(timeout);
    /// <summary>The loaded model, or null for an instance created with <see cref="Wrap"/>.</summary>
    public LlmModel? Model => _model;

    /// <summary>Disposes the engine (and with it the decoder) and the model when this instance owns them.</summary>
    public override void Dispose()
    {
        if (!_ownsEngine)
        {
            return;
        }

        Engine.Dispose();   // stops the scheduler thread and disposes the decoder (device memory)
        _model?.Dispose();
    }
}
