using System.Diagnostics;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tensors;
using Sharplight.Tokenizers;

namespace Sharplight.Llm.Embeddings;

/// <summary>Options for loading an embedding model or a reranker.</summary>
public class PackedModelLoadOptions
{
    /// <summary>Devices in pipeline order (layers are split across them). Required.</summary>
    public required IReadOnlyList<DeviceSlot> Devices { get; init; }
    /// <summary>Fallback device (e.g. CPU) for layers that don't fit and unsupported ops.</summary>
    public DeviceSlot? Fallback { get; init; }
    /// <summary>Most tokens per forward pass (all sequences of a batch together). Longer inputs than this cannot be embedded.</summary>
    public int MaxBatchTokens { get; init; } = 8192;
    /// <summary>Most sequences per forward pass.</summary>
    public int MaxBatchSequences { get; init; } = 256;
    /// <summary>Longest input in tokens (longer inputs are truncated or rejected); 0 = the model's limit.</summary>
    public int MaxSequenceLength { get; init; }
    /// <summary>Device features (<see cref="DeviceFeatures"/>) not to use, e.g. "matrix_f16".</summary>
    public IReadOnlySet<string> DisabledFeatures { get; init; } = new HashSet<string>();
    /// <summary>Options passed to the model adapter.</summary>
    public AdapterOptions Adapter { get; init; } = new();
    /// <summary>Receives load progress messages.</summary>
    public Action<string>? Log { get; init; }
    /// <summary>
    /// Store F32 matrices (rank ≥ 2 weights, e.g. of Hugging Face checkpoints saved in float32) as F16 at load: half the memory and
    /// f16 matrix units on GPUs, at a cosine similarity to the F32 result of ~0.99999. Null (default) = on when the first device is
    /// a GPU; norms, biases and quantized / 16-bit weights are never changed.
    /// </summary>
    public bool? HalfPrecisionMatrices { get; init; }

    internal static (ModelDefinition Def, ModelSource Source, ITokenizer Tokenizer) Open(string path, PackedModelLoadOptions o, AdapterOptions adapter)
    {
        BuiltinAdapters.RegisterAll();
        var sw = Stopwatch.StartNew();
        var sources = ModelImporterRegistry.OpenSet(path);
        if (o.HalfPrecisionMatrices ?? o.Devices.FirstOrDefault()?.Device.Info.Kind is not (null or DeviceKind.Cpu))
        {
            sources = new ModelSourceSet { Primary = HalfMatrices(sources.Primary), Auxiliary = sources.Auxiliary };
        }

        var def = ModelAdapterRegistry.Build(sources, adapter);
        var tok = TokenizerFactory.Create(TokenizerData.FromMetadata(def.SourceMetadata));
        o.Log?.Invoke($"Loaded {def.Name} ({def.Architecture}): {def.Weights.Count} tensors, {def.Weights.Values.Sum(w => w.ByteSize) / 1e9:F2} GB, vocab {tok.VocabSize} [{sw.ElapsedMilliseconds} ms]");
        return (def, sources.Primary, tok);
    }

    private static ModelSource HalfMatrices(ModelSource s)
    {
        if (s.Definition is not null || !s.Weights.Values.Any(w => w.Type == DataType.F32 && w.Shape.Rank >= 2))
        {
            return s;
        }

        var weights = s.Weights.ToDictionary(kv => kv.Key, kv => kv.Value.Type == DataType.F32 && kv.Value.Shape.Rank >= 2 ? ToHalf(kv.Value) : kv.Value);
        return new ModelSource
        {
            Format = s.Format, Architecture = s.Architecture, Metadata = s.Metadata, Weights = weights, OwnedResources = s.OwnedResources,
            FormatData = s.FormatData, Path = s.Path, CompilationCaches = s.CompilationCaches,
        };
    }

    /// <summary>Lazily converted copy (read when the weight is uploaded).</summary>
    private static WeightInfo ToHalf(WeightInfo w) => WeightInfo.Derived(w.Name, DataType.F16, w.Shape, () =>
    {
        var src = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(w.ReadAll());
        var dst = new byte[src.Length * 2];
        var h = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, Half>(dst);
        for (int i = 0; i < src.Length; i++)
        {
            h[i] = (Half)src[i];
        }

        return dst;
    });

    internal int ResolveMaxLength(ModelDefinition def)
    {
        int model = def.Property(WellKnown.ContextLength, 512);
        if (def.SourceMetadata.TryGetValue(CanonicalKeys.TokenizerMaxLength, out var ml))
        {
            model = Math.Min(model, Convert.ToInt32(ml, System.Globalization.CultureInfo.InvariantCulture));
        }

        int limit = MaxSequenceLength > 0 ? MaxSequenceLength : model;
        return Math.Min(limit, MaxBatchTokens);
    }
}
