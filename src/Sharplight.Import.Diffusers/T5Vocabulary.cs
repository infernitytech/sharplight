using System.IO.Compression;
using Sharplight.Import.HuggingFace;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// The T5 v1.1 SentencePiece unigram vocabulary (32 000 pieces with their scores and the 100 sentinel tokens; Apache-2.0), embedded for
/// stand-alone T5 encoder files that come without a tokenizer (<c>t5xxl_fp16.safetensors</c>). It is the <c>tokenizer.json</c> of the
/// FLUX.1 / T5 v1.1 releases without the precompiled normalization map, which the unigram tokenizer does not use.
/// </summary>
internal static class T5Vocabulary
{
    private const string Resource = "Sharplight.Import.Diffusers.t5-tokenizer.json.gz";

    /// <summary>Writes the canonical tokenizer keys of the embedded vocabulary.</summary>
    /// <param name="metadata">The canonical metadata.</param>
    /// <param name="config">The encoder's configuration (special-token fallbacks).</param>
    public static void WriteTo(IDictionary<string, object> metadata, HfConfig config)
    {
        using var stream = typeof(T5Vocabulary).Assembly.GetManifestResourceStream(Resource)
            ?? throw new InvalidOperationException($"Missing embedded resource {Resource}.");
        using var gzip = new GZipStream(stream, CompressionMode.Decompress);
        using var buffer = new MemoryStream();
        gzip.CopyTo(buffer);
        // No companion files: a directory that cannot exist keeps the translator from looking at the current one.
        HfTokenizerTranslator.AddFromJson(metadata, buffer.ToArray(), Path.Combine(AppContext.BaseDirectory, "embedded-t5-vocabulary"), config);
    }
}
