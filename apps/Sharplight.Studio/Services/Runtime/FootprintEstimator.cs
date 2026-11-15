using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Llm.Serving;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Studio.Services.Runtime;

/// <summary>
/// Estimates load footprints with <see cref="LlmModel.EstimateFootprint"/> the way <see cref="Llm.Serving.Hosting.LlmChatModelLoader"/>
/// sizes a load (slots, context, MTP draft head and verification rows). Reading a model builds its graph without touching a device;
/// the last few models read are kept so that changing the context or slots re-estimates at once.
/// </summary>
/// <param name="loader">Reads the models.</param>
internal sealed class FootprintEstimator(ModelLoader loader) : IDisposable
{
    private const int CachedModels = 2;

    private readonly Lock _lock = new();
    private readonly List<(string Key, LlmModel Model)> _models = [];

    /// <summary>The footprint of loading <paramref name="request"/>.</summary>
    /// <param name="request">Model and settings.</param>
    /// <param name="cancellationToken">Cancels reading the model.</param>
    public ModelFootprint Estimate(ModelLoadRequest request, CancellationToken cancellationToken)
    {
        var settings = request.Settings;
        lock (_lock)
        {
            var model = Get(request, cancellationToken);
            int slots = Math.Max(1, settings.Slots);
            bool mtp = settings.MtpDraftTokens > 0 && model.HasDraftHead;
            int drafts = mtp ? settings.MtpDraftTokens : 0;
            int? rows = drafts > 0 ? slots + (Math.Min(slots, new EngineOptions().SpeculationMaxBatch) * drafts) : null;
            return model.EstimateFootprint(slots, Math.Max(256, settings.ContextLength), null, drafts, mtp, rows);
        }
    }

    /// <summary>Closes the models read.</summary>
    public void Dispose()
    {
        lock (_lock)
        {
            foreach (var (_, model) in _models)
            {
                model.Dispose();
            }

            _models.Clear();
        }
    }

    /// <summary>The model read with the settings that change its graph (KV cache type, precision, batch), most recent first.</summary>
    private LlmModel Get(ModelLoadRequest request, CancellationToken cancellationToken)
    {
        var settings = request.Settings;
        var key = string.Join('\n', request.Path, settings.KvCacheType, settings.AllowReducedPrecision, settings.BatchTokens);
        int index = _models.FindIndex(entry => entry.Key == key);
        if (index >= 0)
        {
            var hit = _models[index];
            _models.RemoveAt(index);
            _models.Insert(0, hit);
            return hit.Model;
        }

        var model = LlmModel.Load(loader, request.Path, new LlmLoadOptions
        {
            Devices = [],
            ContextLength = Math.Max(256, settings.ContextLength),
            MaxBatchTokens = Math.Max(16, settings.BatchTokens),
            KvCacheType = DataType.Get(settings.KvCacheType),
            AllowReducedPrecision = settings.AllowReducedPrecision,
            CancellationToken = cancellationToken,
        });
        _models.Insert(0, (key, model));
        while (_models.Count > CachedModels)
        {
            _models[^1].Model.Dispose();
            _models.RemoveAt(_models.Count - 1);
        }

        return model;
    }
}
