using Sharplight.Llm.Serving.Chat;

namespace Sharplight.Llm.Serving.Hosting;

/// <summary>A loaded chat model: weights on the devices, a batch decoder, its own scheduler thread (<see cref="Engine"/>) and chat layer.</summary>
public sealed class ChatModelInstance : LoadedModel, IChatModel
{
    private readonly LlmModel? _model;
    private readonly bool _ownsEngine;
    public InferenceEngine Engine { get; }
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

    public override ModelKind Kind => ModelKind.Chat;
    public override long ResidentBytes => _resident;
    public override bool IsIdle { get { var m = Engine.GetMetrics(); return m.ActiveSequences == 0 && m.QueueDepth == 0; } }
    public override Task<bool> DrainAsync(TimeSpan timeout) => Engine.DrainAsync(timeout);
    public LlmModel? Model => _model;

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
