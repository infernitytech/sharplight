using Sharplight.Llm.Serving.Chat;
using Sharplight.Models;
using Sharplight.Runtime;

namespace Sharplight.Llm.Serving.Hosting;

/// <summary>Loads a chat model file (any format with an importer registered) onto devices shared with other models.</summary>
/// <param name="loader">Opens and builds the model file.</param>
/// <param name="name">Model name reported by the APIs.</param>
/// <param name="path">Model file or directory.</param>
/// <param name="load">Devices and compile settings; the devices may be shared with other loaders.</param>
/// <param name="chat">Slots, engine and chat-service settings.</param>
public sealed class LlmChatModelLoader(ModelLoader loader, string name, string path, LlmLoadOptions load, ChatModelOptions? chat = null) : IHostedModelLoader
{
    private readonly ChatModelOptions _chat = chat ?? new();
    /// <inheritdoc/>
    public ModelKind Kind => ModelKind.Chat;
    /// <summary>The model file or directory.</summary>
    public string Path => path;

    /// <inheritdoc/>
    public IPreparedModel Prepare(CancellationToken ct)
    {
        var model = LlmModel.Load(loader, path, load);
        try
        {
            ct.ThrowIfCancellationRequested();
            // Speculation with the model's MTP head: rollback records, verification logits rows and the head itself. (A custom
            // speculator's draft count is known only once it is created at load; its records are not reserved here.)
            bool mtp = _chat.MtpDraftTokens > 0 && model.HasDraftHead;
            int drafts = mtp ? _chat.MtpDraftTokens : 0;
            int? rows = drafts > 0 ? _chat.Slots + Math.Min(_chat.Slots, _chat.Engine.SpeculationMaxBatch) * drafts : null;
            // Image input: the multimodal decode graph and the vision encoder, loaded with the model and counted in its footprint.
            bool media = _chat.ImageInput && model.SupportsImages && model.SupportsMediaInput && model.Architectures.TryFindMultimodal(model.Definition.Architecture, out _);
            var fp = model.EstimateFootprint(_chat.Slots, load.ContextLength, _chat.KvCacheTokens, drafts, mtp, rows, _chat.KvStateSnapshots, media);
            long headBytes = mtp ? fp.Total - model.EstimateFootprint(_chat.Slots, load.ContextLength, _chat.KvCacheTokens, drafts, false, rows, _chat.KvStateSnapshots, media).Total : 0;
            long visionBytes = media ? model.EstimateVisionFootprint().Total : 0;
            // The model is split over the pipeline devices in proportion to their weights (equal when unset).
            var devs = load.Devices;
            double sum = devs.Sum(d => d.Weight ?? 1);
            var reqs = devs.Select(d => new MemoryRequirement(new DeviceMemoryDomain(d.Device), (long)(fp.Total * ((d.Weight ?? 1) / sum)))).ToList();
            return new Prepared(this, model, fp, headBytes, visionBytes, media, reqs);
        }
        catch
        {
            model.Dispose();
            throw;
        }
    }

    private sealed class Prepared(LlmChatModelLoader owner, LlmModel model, ModelFootprint fp, long headBytes, long visionBytes, bool media,
        IReadOnlyList<MemoryRequirement> reqs) : IPreparedModel
    {
        private bool _consumed;
        public IReadOnlyList<MemoryRequirement> Requirements => reqs;
        public bool CanOffload => owner.Fallback is not null;

        /// <summary>
        /// With a default-sized paged KV cache on one device: the footprint with the smallest cache <see cref="KvTokensFor"/> shrinks
        /// to (one full context plus a block per slot).
        /// </summary>
        public long? ShrinkableTo
        {
            get
            {
                if (owner._chat.KvCacheTokens is not null || reqs.Count != 1 || DefaultKvCache() is not { } kv)
                {
                    return null;
                }

                return fp.Total - (kv.Tokens - kv.Least) * kv.PerToken;
            }
        }

        /// <summary>The default paged cache's tokens, the fewest it may shrink to and its bytes per token; null without one.</summary>
        private (long Tokens, long Least, long PerToken)? DefaultKvCache()
        {
            int ctx = owner.Load.ContextLength;
            long tokens = model.DefaultKvCacheTokens(owner._chat.Slots, ctx);
            long perToken = model.KvBytesPerToken;
            if (tokens <= 0 || perToken <= 0)
            {
                return null;
            }

            return (tokens, Math.Min(tokens, ctx + (long)owner._chat.Slots * MultiSequenceDecoder.DefaultKvBlockSize), perToken);
        }
        public ModelFootprint Footprint => fp;

        public LoadedModel Load(long? primaryLimit)
        {
            // The limit covers the whole model on the primary device; placement only sees weights + state, so leave room for activations.
            // The draft head (compiled first) and the vision encoder take their shares of the limit too.
            long? limit = primaryLimit is { } l ? Math.Max(0, l - fp.ActivationBytes - headBytes - visionBytes) : null;
            var chat = owner._chat;
            long? kvTokens = KvTokensFor(primaryLimit);
            var speculator = chat.MtpDraftTokens > 0 && model.HasDraftHead
                ? model.CreateMtpSpeculator(chat.Slots, chat.MtpDraftTokens, owner.Load.ContextLength, kvTokens, media)
                : chat.Speculator?.Invoke(model);
            var options = owner._chat.Engine with { Speculator = speculator ?? owner._chat.Engine.Speculator };
            int? logitsRows = options.Speculator is { } spec
                ? owner._chat.Slots + Math.Min(owner._chat.Slots, options.SpeculationMaxBatch) * spec.MaxDraftTokens
                : null;
            IBatchDecoder decoder;
            ImagePrompts? images = null;
            try
            {
                decoder = model.CreateBatchDecoder(owner._chat.Slots, owner.Load.ContextLength, limit, logitsRows, kvTokens,
                    options.Speculator?.MaxDraftTokens ?? 0, chat.KvStateSnapshots, media);
            }
            catch
            {
                speculator?.Dispose();
                throw;
            }

            InferenceEngine? engine = null;
            try
            {
                // The vision encoder before the engine starts its scheduler thread (which prepares the decode steps on the device).
                images = media && decoder.EmbeddingWidth > 0 ? new ImagePrompts(model.CreateVisionEncoder(), model.Tokenizer, ownsEncoder: true) : null;
                engine = new InferenceEngine(decoder, model.Tokenizer, options);
                var service = new ModelService(engine, model.ChatTemplate, owner._chat.Service with { ModelName = owner.ModelName }, images);
                _consumed = true;   // the instance owns the model now
                long head = speculator?.DeviceMemoryBytes ?? 0;
                return new ChatModelInstance(model, engine, service, decoder.DeviceMemoryBytes + head + (images?.DeviceMemoryBytes ?? 0), images: images);
            }
            catch
            {
                images?.Dispose();
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

        /// <summary>
        /// The paged cache size: the configured one, or by default every slot's full context, shrunk (to at least one full context) when the
        /// host loads the model into <paramref name="primaryLimit"/> bytes, less than it needs: the cache gives way before layers are offloaded.
        /// </summary>
        private long? KvTokensFor(long? primaryLimit)
        {
            var chat = owner._chat;
            if (chat.KvCacheTokens is not null || primaryLimit is not { } limit || limit >= fp.Total)
            {
                return chat.KvCacheTokens;
            }

            if (DefaultKvCache() is not { } kv)
            {
                return null;
            }

            var (tokens, least, perToken) = kv;
            long fit = (limit - (fp.Total - tokens * perToken)) / perToken;
            long shrunk = Math.Clamp(fit, least, tokens);
            if (shrunk < tokens)
            {
                owner.Load.Log?.Invoke($"Paged KV cache shrunk to {shrunk} tokens ({shrunk * perToken / 1048576.0:F0} MiB) to fit {limit / 1048576.0:F0} MiB.");
            }

            return shrunk;
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
