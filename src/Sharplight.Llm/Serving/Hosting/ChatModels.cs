using Sharplight.Llm.Serving.Chat;
using Sharplight.Runtime;

namespace Sharplight.Llm.Serving.Hosting;

/// <summary>How a chat model is served: batching slots, engine and chat-service settings.</summary>
public sealed record ChatModelOptions
{
    /// <summary>Concurrent sequences batched per forward pass, each with the load option's context length.</summary>
    public int Slots { get; init; } = 4;
    public EngineOptions Engine { get; init; } = new();
    /// <summary>Name reported by the API is the registration name; the other settings (chat template override, tool formats, default max tokens) apply.</summary>
    public ModelServiceOptions Service { get; init; } = new();
}

/// <summary>A loaded model that serves chat/completion requests.</summary>
public interface IChatModel
{
    public InferenceEngine Engine { get; }
    public ModelService Service { get; }
}

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
        _model = model; Engine = engine; Service = service; _resident = resident; _ownsEngine = ownsEngine;
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

/// <summary>Loads a chat model file (any format with an importer registered) onto devices shared with other models.</summary>
public sealed class LlmChatModelLoader(string name, string path, LlmLoadOptions load, ChatModelOptions? chat = null) : IModelLoader
{
    private readonly ChatModelOptions _chat = chat ?? new();
    public ModelKind Kind => ModelKind.Chat;
    public string Path => path;

    public IPreparedModel Prepare(CancellationToken ct)
    {
        var model = LlmModel.Load(path, load);
        try
        {
            var fp = model.EstimateFootprint(_chat.Slots, load.ContextLength);
            // The model is split over the pipeline devices in proportion to their weights (equal when unset).
            var devs = load.Devices;
            double sum = devs.Sum(d => d.Weight ?? 1);
            var reqs = devs.Select(d => new MemoryRequirement(new DeviceMemoryDomain(d.Device), (long)(fp.Total * ((d.Weight ?? 1) / sum)))).ToList();
            return new Prepared(this, model, fp, reqs);
        }
        catch { model.Dispose(); throw; }
    }

    private sealed class Prepared(LlmChatModelLoader owner, LlmModel model, ModelFootprint fp, IReadOnlyList<MemoryRequirement> reqs) : IPreparedModel
    {
        private bool _consumed;
        public IReadOnlyList<MemoryRequirement> Requirements => reqs;
        public bool CanOffload => owner.Fallback is not null;
        public ModelFootprint Footprint => fp;

        public LoadedModel Load(long? primaryLimit)
        {
            // The limit covers the whole model on the primary device; placement only sees weights + state, so leave room for activations.
            long? limit = primaryLimit is { } l ? Math.Max(0, l - fp.ActivationBytes) : null;
            var decoder = model.CreateBatchDecoder(owner._chat.Slots, owner.Load.ContextLength, limit);
            InferenceEngine? engine = null;
            try
            {
                engine = new InferenceEngine(decoder, model.Tokenizer, owner._chat.Engine);
                var service = new ModelService(engine, model.ChatTemplate, owner._chat.Service with { ModelName = owner.ModelName });
                _consumed = true;   // the instance owns the model now
                return new ChatModelInstance(model, engine, service, decoder.DeviceMemoryBytes);
            }
            catch
            {
                if (engine is null)
                {
                    decoder.Dispose();
                }
                else
                {
                    engine.Dispose();
                }

                throw;
            }
        }

        public void Dispose() { if (!_consumed)
            {
                model.Dispose();
            }
        }
    }

    private DeviceSlot? Fallback => load.Fallback;
    private LlmLoadOptions Load => load;
    private string ModelName => name;
}
