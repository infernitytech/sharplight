using Sharplight.Llm.Serving.Chat;
using Sharplight.Runtime;

namespace Sharplight.Llm.Serving.Hosting;

/// <summary>Loads a chat model file (any format with an importer registered) onto devices shared with other models.</summary>
/// <param name="name">Model name reported by the APIs.</param>
/// <param name="path">Model file or directory.</param>
/// <param name="load">Devices and compile settings; the devices may be shared with other loaders.</param>
/// <param name="chat">Slots, engine and chat-service settings.</param>
public sealed class LlmChatModelLoader(string name, string path, LlmLoadOptions load, ChatModelOptions? chat = null) : IModelLoader
{
    private readonly ChatModelOptions _chat = chat ?? new();
    /// <inheritdoc/>
    public ModelKind Kind => ModelKind.Chat;
    /// <summary>The model file or directory.</summary>
    public string Path => path;

    /// <inheritdoc/>
    public IPreparedModel Prepare(CancellationToken ct)
    {
        var model = LlmModel.Load(path, load);
        try
        {
            ct.ThrowIfCancellationRequested();
            // Speculation with the model's MTP head: rollback records, verification logits rows and the head itself. (A custom
            // speculator's draft count is known only once it is created at load; its records are not reserved here.)
            bool mtp = _chat.MtpDraftTokens > 0 && model.HasDraftHead;
            int drafts = mtp ? _chat.MtpDraftTokens : 0;
            int? rows = drafts > 0 ? _chat.Slots + Math.Min(_chat.Slots, _chat.Engine.SpeculationMaxBatch) * drafts : null;
            var fp = model.EstimateFootprint(_chat.Slots, load.ContextLength, _chat.KvCacheTokens, drafts, mtp, rows);
            long headBytes = mtp ? fp.Total - model.EstimateFootprint(_chat.Slots, load.ContextLength, _chat.KvCacheTokens, drafts, false, rows).Total : 0;
            // The model is split over the pipeline devices in proportion to their weights (equal when unset).
            var devs = load.Devices;
            double sum = devs.Sum(d => d.Weight ?? 1);
            var reqs = devs.Select(d => new MemoryRequirement(new DeviceMemoryDomain(d.Device), (long)(fp.Total * ((d.Weight ?? 1) / sum)))).ToList();
            return new Prepared(this, model, fp, headBytes, reqs);
        }
        catch
        {
            model.Dispose();
            throw;
        }
    }

    private sealed class Prepared(LlmChatModelLoader owner, LlmModel model, ModelFootprint fp, long headBytes, IReadOnlyList<MemoryRequirement> reqs) : IPreparedModel
    {
        private bool _consumed;
        public IReadOnlyList<MemoryRequirement> Requirements => reqs;
        public bool CanOffload => owner.Fallback is not null;
        public ModelFootprint Footprint => fp;

        public LoadedModel Load(long? primaryLimit)
        {
            // The limit covers the whole model on the primary device; placement only sees weights + state, so leave room for activations.
            // The draft head (compiled first) takes its share of the limit too.
            long? limit = primaryLimit is { } l ? Math.Max(0, l - fp.ActivationBytes - headBytes) : null;
            var chat = owner._chat;
            var speculator = chat.MtpDraftTokens > 0 && model.HasDraftHead
                ? model.CreateMtpSpeculator(chat.Slots, chat.MtpDraftTokens, owner.Load.ContextLength, chat.KvCacheTokens)
                : chat.Speculator?.Invoke(model);
            var options = owner._chat.Engine with { Speculator = speculator ?? owner._chat.Engine.Speculator };
            int? logitsRows = options.Speculator is { } spec
                ? owner._chat.Slots + Math.Min(owner._chat.Slots, options.SpeculationMaxBatch) * spec.MaxDraftTokens
                : null;
            IBatchDecoder decoder;
            try
            {
                decoder = model.CreateBatchDecoder(owner._chat.Slots, owner.Load.ContextLength, limit, logitsRows, owner._chat.KvCacheTokens,
                    options.Speculator?.MaxDraftTokens ?? 0);
            }
            catch
            {
                speculator?.Dispose();
                throw;
            }

            InferenceEngine? engine = null;
            try
            {
                engine = new InferenceEngine(decoder, model.Tokenizer, options);
                var service = new ModelService(engine, model.ChatTemplate, owner._chat.Service with { ModelName = owner.ModelName });
                _consumed = true;   // the instance owns the model now
                long head = speculator is Speculative.MtpSpeculator mtp ? mtp.DeviceMemoryBytes : 0;
                return new ChatModelInstance(model, engine, service, decoder.DeviceMemoryBytes + head);
            }
            catch
            {
                if (engine is null)
                {
                    decoder.Dispose();
                    speculator?.Dispose();
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
