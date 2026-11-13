using Sharplight.Llm.Serving.Chat;
using Sharplight.Runtime;

namespace Sharplight.Llm.Serving.Hosting;

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
        catch
        {
            model.Dispose();
            throw;
        }
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
            long? limit = primaryLimit is { }
            l ? Math.Max(0, l - fp.ActivationBytes) : null;
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

        public void Dispose()
        {
            if (!_consumed)
            {
                model.Dispose();
            }
        }
    }

    private DeviceSlot? Fallback => load.Fallback;
    private LlmLoadOptions Load => load;
    private string ModelName => name;
}
