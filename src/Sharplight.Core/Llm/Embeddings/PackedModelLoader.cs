using System.Diagnostics;
using System.Runtime.InteropServices;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tensors;
using Sharplight.Tokenizers;

namespace Sharplight.Llm.Embeddings;

/// <summary>Opens the weights and tokenizer of an embedding model or reranker.</summary>
internal static class PackedModelLoader
{
    /// <summary>Opens the model files at <paramref name="path"/>, builds its definition and creates its tokenizer.</summary>
    internal static (ModelDefinition Def, ModelSource Source, ITokenizer Tokenizer) Open(string path, PackedModelLoadOptions o, AdapterOptions adapter)
    {
        var sw = Stopwatch.StartNew();
        var sources = ModelImporterRegistry.OpenSet(path);
        if (o.HalfPrecisionMatrices ?? o.Devices.FirstOrDefault()?.Device.Info.Kind is not (null or DeviceKind.Cpu))
        {
            sources = new ModelSourceSet { Primary = HalfMatrices(sources.Primary), Auxiliary = sources.Auxiliary };
        }

        var def = ModelAdapterRegistry.Build(sources, adapter, o.Log);
        var tok = TokenizerFactory.Create(TokenizerData.FromMetadata(def.SourceMetadata));
        o.Log?.Invoke($"Loaded {def.Name} ({def.Architecture}): {def.Weights.Count} tensors, {def.Weights.Values.Sum(w => w.ByteSize) / 1e9:F2} GB, vocab {tok.VocabSize} [{sw.ElapsedMilliseconds} ms]");
        return (def, sources.Primary, tok);
    }

    private static ModelSource HalfMatrices(ModelSource s)
    {
        if (s.Definition is not null || !s.Weights.Values.Any(IsHalved))
        {
            return s;
        }

        var weights = s.Weights.ToDictionary(kv => kv.Key, kv => IsHalved(kv.Value) ? ToHalf(kv.Value) : kv.Value);
        return new ModelSource
        {
            Format = s.Format, Architecture = s.Architecture, Metadata = s.Metadata, Weights = weights, OwnedResources = s.OwnedResources,
            FormatData = s.FormatData, Path = s.Path, CompilationCaches = s.CompilationCaches,
        };
    }

    // F32 matrices, except the short-conv kernels of recurrent layers (decision models on hybrid trunks), which backends take in f32.
    private static bool IsHalved(WeightInfo w) => w.Type == DataType.F32 && w.Shape.Rank >= 2 && !w.Name.EndsWith(".conv1d.weight", StringComparison.Ordinal);

    /// <summary>Lazily converted copy (read when the weight is uploaded).</summary>
    private static WeightInfo ToHalf(WeightInfo w) => WeightInfo.Derived(w.Name, DataType.F16, w.Shape, () =>
    {
        var src = MemoryMarshal.Cast<byte, float>(w.ReadAll());
        var dst = new byte[src.Length * 2];
        var h = MemoryMarshal.Cast<byte, Half>(dst);
        for (int i = 0; i < src.Length; i++)
        {
            h[i] = (Half)src[i];
        }

        return dst;
    });
}
