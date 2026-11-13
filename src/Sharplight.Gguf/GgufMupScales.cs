namespace Sharplight.Gguf;

/// <summary>muP-style multipliers (Granite, MiniCPM) in llama.cpp's convention, translated to the canonical scale keys.</summary>
internal static class GgufMupScales
{
    /// <summary>
    /// Adds <c>embedding_scale</c> (token embeddings are multiplied), <c>residual.scale</c> (every sub-layer output is multiplied before the
    /// residual add), <c>attention.scale</c> (replaces 1/sqrt(head_dim)) and <c>logit_scale</c>. llama.cpp stores the logit scale as a
    /// divisor, the canonical key is the multiplier. When the file lacks a key the given default applies (null = key left out).
    /// </summary>
    public static void Add(Dictionary<string, object> md, GgufMetadataView g, float? embeddingScale = null, float? residualScale = null, float? logitDivisor = null)
    {
        string prefix = GgufDecoderHyperParameters.Prefix;
        if (Read(g, "embedding_scale", embeddingScale) is { } embedding)
        {
            md[prefix + "embedding_scale"] = embedding;
        }

        if (Read(g, "residual_scale", residualScale) is { } residual)
        {
            md[prefix + "residual.scale"] = residual;
        }

        if (g.Has("attention.scale"))
        {
            md[prefix + "attention.scale"] = g.Float("attention.scale");
        }

        if (Read(g, "logit_scale", logitDivisor) is { } divisor && divisor != 0f)
        {
            md[prefix + "logit_scale"] = 1f / divisor;
        }
    }

    private static float? Read(GgufMetadataView g, string key, float? fallback) => g.Has(key) ? g.Float(key) : fallback;
}
