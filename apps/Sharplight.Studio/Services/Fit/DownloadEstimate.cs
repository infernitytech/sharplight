using Sharplight.Runtime;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Hub;

namespace Sharplight.Studio.Services.Fit;

/// <summary>
/// A conservative footprint of a model that is not downloaded yet: the download size as weights, the KV cache from the dimensions
/// when they could be read (else a generous share of the weights), and an activation allowance. Only an approximation of what
/// <see cref="Runtime.IModelRuntime.EstimateFootprintAsync"/> computes once the model is on disk.
/// </summary>
internal static class DownloadEstimate
{
    private const long MiB = 1L << 20;

    /// <summary>The context the estimate assumes: the default load context, at most what the model was trained for.</summary>
    /// <param name="shape">The dimensions, when known.</param>
    /// <param name="settings">The default load settings.</param>
    public static int Context(ModelShape? shape, LoadSettings settings)
        => Math.Max(256, Math.Min(settings.ContextLength, shape?.ContextLength ?? int.MaxValue));

    /// <summary>The estimated footprint.</summary>
    /// <param name="downloadBytes">The variant's download size (weights, plus a vision projector when it has one).</param>
    /// <param name="shape">The dimensions, when they could be read.</param>
    /// <param name="settings">The default load settings (context, slots, KV cache type, batch).</param>
    public static ModelFootprint Estimate(long downloadBytes, ModelShape? shape, LoadSettings settings)
    {
        long weights = Math.Max(0, downloadBytes);
        int context = Context(shape, settings);
        int slots = Math.Max(1, settings.Slots);
        long kv;
        if (shape?.Layers is { } layers && shape.KvElementsPerTokenLayer is { } perTokenLayer)
        {
            kv = (long)(layers * perTokenLayer * (double)context * slots * KvBytesPerElement(settings.KvCacheType));
        }
        else
        {
            // Unknown dimensions: a tenth of the weights per 8k tokens of context and slot, at least 256 MiB.
            kv = (long)(Math.Max(256 * MiB, weights * 0.1) * (context / 8192.0) * slots);
        }

        long batch = Math.Max(16, settings.BatchTokens);
        long hidden = shape?.HiddenSize is { } size
            ? batch * size * sizeof(float) * 24
            : (long)(weights * 0.02);
        long logits = shape?.VocabularySize is { } vocabulary ? (long)vocabulary * slots * sizeof(float) * 2 : 64 * MiB;
        return new ModelFootprint(weights, kv, (256 * MiB) + hidden + logits);
    }

    /// <summary>Bytes per KV cache element of a cache type (block-quantized types with their scales).</summary>
    /// <param name="type">The KV cache type name.</param>
    public static double KvBytesPerElement(string type) => type switch
    {
        "f32" => 4,
        "q8_0" => 34.0 / 32,
        "f8e4m3_b32" => 36.0 / 32,
        "q4_0" => 18.0 / 32,
        _ => 2,
    };
}
