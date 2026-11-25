using System.Diagnostics;
using Sharplight.Models;
using Sharplight.Tokenizers;

namespace Sharplight.Llm.Embeddings;

/// <summary>Opens the weights and tokenizer of a packed-sequence model (span scorer).</summary>
internal static class PackedModelLoader
{
    /// <summary>Opens the model files at <paramref name="path"/>, builds its definition and creates its tokenizer.</summary>
    internal static (ModelDefinition Def, ModelSource Source, ITokenizer Tokenizer) Open(ModelLoader loader, string path, PackedModelLoadOptions o, AdapterOptions adapter)
    {
        ArgumentNullException.ThrowIfNull(loader);
        ArgumentNullException.ThrowIfNull(o);
        if (o.Devices.Count == 0)
        {
            throw new ArgumentException("Devices is empty: give at least one device.", nameof(o));
        }

        var sw = Stopwatch.StartNew();
        bool halve = HalfPrecisionMatrices.Enabled(o.HalfPrecisionMatrices, o.Devices);
        var (def, source) = loader.Load(path, [], adapter, progress: null, o.Log, CancellationToken.None, halve ? HalfPrecisionMatrices.Transform : null);
        var tok = TokenizerFactory.Create(TokenizerData.FromMetadata(def.SourceMetadata));
        o.Log?.Invoke($"Loaded {def.Name} ({def.Architecture}): {def.Weights.Count} tensors, {def.Weights.Values.Sum(w => w.ByteSize) / 1e9:F2} GB, vocab {tok.VocabSize} [{sw.ElapsedMilliseconds} ms]");
        return (def, source, tok);
    }
}
