using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Models;

/// <summary>Provides the raw bytes of a weight tensor (e.g. a memory-mapped file region).</summary>
public interface IWeightSource
{
    /// <summary>Size of the stored data in bytes.</summary>
    long ByteLength { get; }

    /// <summary>Copies <paramref name="destination"/>.Length bytes starting at <paramref name="offset"/>.</summary>
    void Read(long offset, Span<byte> destination);

    /// <summary>
    /// Returns a pointer to the data if it is directly addressable (memory mapped), enabling zero-copy
    /// uploads; otherwise <see cref="IntPtr.Zero"/>.
    /// </summary>
    nint TryGetPointer() => 0;
}

/// <summary>A model parameter: name, logical shape, storage type and where its bytes live.</summary>
public sealed class WeightInfo
{
    public required string Name { get; init; }
    public required DataType Type { get; init; }
    /// <summary>Logical shape outermost-first, e.g. a linear layer is [N_out, K_in].</summary>
    public required Shape Shape { get; init; }
    public required IWeightSource Source { get; init; }
    public long ByteSize => Type.ByteSize(Shape.Elements());
    public override string ToString() => $"{Name} {Type}{Shape}";

    /// <summary>
    /// A weight whose bytes are computed on first access (e.g. an importer converting a sign
    /// convention, folding a constant or re-laying out data). The producer must return exactly
    /// <c>type.ByteSize(shape.Elements())</c> bytes.
    /// </summary>
    public static WeightInfo Derived(string name, DataType type, Shape shape, Func<byte[]> produce)
        => new() { Name = name, Type = type, Shape = shape, Source = new DerivedWeightSource(type.ByteSize(shape.Elements()), produce) };

    /// <summary>Same bytes under a different name and/or shape (element count must match).</summary>
    public WeightInfo As(string name, Shape? shape = null)
    {
        var s = shape ?? Shape;
        if (s.Elements() != Shape.Elements()) throw new ArgumentException($"{Name}: cannot view {Shape} as {s}.");
        return new() { Name = name, Type = Type, Shape = s, Source = Source };
    }

    /// <summary>Reads the whole tensor.</summary>
    public byte[] ReadAll()
    {
        var b = new byte[Source.ByteLength];
        Source.Read(0, b);
        return b;
    }
}

sealed class DerivedWeightSource(long length, Func<byte[]> produce) : IWeightSource
{
    readonly Lazy<byte[]> _data = new(() =>
    {
        var d = produce();
        if (d.LongLength != length) throw new InvalidDataException($"Derived weight produced {d.LongLength} bytes, expected {length}.");
        return d;
    });
    public long ByteLength => length;
    public void Read(long offset, Span<byte> destination) => _data.Value.AsSpan(checked((int)offset), destination.Length).CopyTo(destination);
}

/// <summary>Kind of persistent state; lets runtimes manage e.g. context shifting or snapshots per kind.</summary>
public enum StateKind { KeyCache, ValueCache, Recurrent, Other }

/// <summary>
/// Persistent per-sequence tensor (KV cache, recurrent/conv state). The outermost dimension of a
/// cache may be the symbolic context length <c>ctx</c>, bound when the model is compiled.
/// </summary>
public sealed class StateDefinition
{
    public required string Name { get; init; }
    public required DataType Type { get; init; }
    public required Shape Shape { get; init; }
    public StateKind Kind { get; init; } = StateKind.Other;
    /// <summary>Placement group the state belongs to (same group as the layer using it).</summary>
    public string Group { get; init; } = "";
    public override string ToString() => $"{Name} {Type}{Shape} ({Kind})";
}

/// <summary>Semantic role of a graph input so generic runtimes (LLM sessions, vision pipelines) can feed it.</summary>
public enum InputRole
{
    TokenIds,
    /// <summary>Position ids, shape [axes, T].</summary>
    Positions,
    /// <summary>Precomputed embeddings injected instead of token embeddings (multimodal).</summary>
    Embeddings,
    /// <summary>Indices of rows for which outputs (logits) are requested.</summary>
    OutputRows,
    /// <summary>Image / audio pixel or feature data.</summary>
    Media,
    Other,
    /// <summary>
    /// Per-row selector [T, 1] f32 of a graph that takes both token ids and embeddings: 1 = take the row from the
    /// embeddings input, 0 = from the token embedding table (multimodal prompts mixing text and media).
    /// Declared last so the stored values of the other roles (saved packages) stay valid.
    /// </summary>
    EmbeddingMask,
}

public sealed class InputDefinition
{
    public required string Name { get; init; }
    public required InputRole Role { get; init; }
    public override string ToString() => $"{Name} ({Role})";
}

/// <summary>
/// The format-agnostic result of importing a model: weights, persistent state definitions and one
/// or more named compute graphs (e.g. <c>"decode"</c>, <c>"vision_encode"</c>, <c>"mtp_draft"</c>).
/// </summary>
public sealed class ModelDefinition
{
    public required string Architecture { get; init; }
    public string Name { get; init; } = "";
    public required IReadOnlyDictionary<string, WeightInfo> Weights { get; init; }
    public required IReadOnlyDictionary<string, ComputeGraph> Graphs { get; init; }
    public IReadOnlyList<StateDefinition> States { get; init; } = [];

    /// <summary>Architecture-level facts runtimes may need (vocab size, context length, eos ids, ...).</summary>
    public IReadOnlyDictionary<string, object> Properties { get; init; } = new Dictionary<string, object>();

    /// <summary>Canonical source metadata (<see cref="CanonicalKeys"/>) for consumers such as tokenizers.</summary>
    public IReadOnlyDictionary<string, object> SourceMetadata { get; init; } = new Dictionary<string, object>();

    /// <summary>Resources that must be disposed with the definition (memory maps, file handles).</summary>
    public IReadOnlyList<IDisposable> OwnedResources { get; init; } = [];

    public T Property<T>(string key) => (T)Convert.ChangeType(Properties[key], typeof(T), System.Globalization.CultureInfo.InvariantCulture);
    public T Property<T>(string key, T defaultValue) => Properties.TryGetValue(key, out var v)
        ? (T)Convert.ChangeType(v, typeof(T), System.Globalization.CultureInfo.InvariantCulture) : defaultValue;

    public ComputeGraph Graph(string name) => Graphs.TryGetValue(name, out var g)
        ? g : throw new KeyNotFoundException($"Model '{Architecture}' has no graph '{name}'. Available: {string.Join(", ", Graphs.Keys)}");
}

/// <summary>Well-known graph and property names shared by adapters and runtimes.</summary>
public static class WellKnown
{
    public const string DecodeGraph = "decode";
    public const string VisionEncodeGraph = "vision_encode";
    /// <summary>Decoder taking token ids, an embeddings input and a per-row mask (<see cref="InputRole.EmbeddingMask"/>): text mixed with media embeddings.</summary>
    public const string DecodeMultimodalGraph = "decode_multimodal";
    public const string TokenSymbol = "T";
    public const string ContextSymbol = "ctx";
    public const string PastScalar = "n_past";

    public const string VocabSize = "vocab_size";
    public const string ContextLength = "context_length";
    public const string EmbeddingLength = "embedding_length";
    public const string PositionAxes = "position_axes";
    public const string EosTokenIds = "eos_token_ids";
}
