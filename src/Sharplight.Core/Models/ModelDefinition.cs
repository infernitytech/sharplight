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

    /// <summary>
    /// Zero-copy view of rows [firstRow, firstRow + rows) of a tensor viewed as [Shape[0], rest...] (e.g. splitting a fused
    /// q/k/v or gate/up projection). Rows are contiguous in every storage type, including block-quantized ones.
    /// </summary>
    public WeightInfo RowSlice(string name, long firstRow, long rows)
    {
        long total = Shape[0].Fixed;
        if (firstRow < 0 || rows <= 0 || firstRow + rows > total) throw new ArgumentOutOfRangeException(nameof(rows), $"{Name}: rows [{firstRow}, {firstRow + rows}) outside [0, {total}).");
        long rowElems = Shape.Elements() / total;
        long rowBytes = Type.ByteSize(rowElems);
        var dims = new List<Dim> { rows };
        for (int d = 1; d < Shape.Rank; d++) dims.Add(Shape[d]);
        return new() { Name = name, Type = Type, Shape = new Shape(dims), Source = new SliceWeightSource(Source, firstRow * rowBytes, rows * rowBytes) };
    }

    /// <summary>
    /// Stacks equally shaped tensors along a new outermost axis (per-expert matrices into one [E, N, K] tensor). The result reads
    /// its parts lazily; nothing is copied into memory up front.
    /// </summary>
    public static WeightInfo Stack(string name, IReadOnlyList<WeightInfo> parts)
    {
        var first = parts[0];
        foreach (var w in parts)
            if (w.Type != first.Type || !w.Shape.Equals(first.Shape)) throw new ArgumentException($"{name}: cannot stack {w} with {first}.");
        var dims = new List<Dim> { parts.Count };
        for (int d = 0; d < first.Shape.Rank; d++) dims.Add(first.Shape[d]);
        return new() { Name = name, Type = first.Type, Shape = new Shape(dims), Source = new StackedWeightSource([.. parts.Select(w => w.Source)]) };
    }

    /// <summary>Reads the whole tensor.</summary>
    public byte[] ReadAll()
    {
        var b = new byte[Source.ByteLength];
        Source.Read(0, b);
        return b;
    }
}

sealed class StackedWeightSource(IWeightSource[] parts) : IWeightSource
{
    readonly long _each = parts[0].ByteLength;
    public long ByteLength => _each * parts.Length;
    public void Read(long offset, Span<byte> destination)
    {
        while (destination.Length > 0)
        {
            int i = (int)(offset / _each);
            long inPart = offset - i * _each;
            int n = (int)Math.Min(destination.Length, _each - inPart);
            parts[i].Read(inPart, destination[..n]);
            destination = destination[n..];
            offset += n;
        }
    }
}

sealed class SliceWeightSource(IWeightSource parent, long offset, long length) : IWeightSource
{
    public long ByteLength => length;
    public void Read(long off, Span<byte> destination) => parent.Read(offset + off, destination);
    public nint TryGetPointer() => parent.TryGetPointer() is var p && p != 0 ? p + (nint)offset : 0;
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
    /// <summary>
    /// Multi-sequence graphs (<see cref="Graph.MultiSequence"/>): per-row table [T, 4] i32 of (slot, position in the slot's
    /// context, row count of the row's run, run index).
    /// </summary>
    SequenceRows,
    /// <summary>Multi-sequence graphs: per-run table [U, 4] i32 of (slot, first row, row count, first position); a run is a
    /// slot's consecutive rows, in context order.</summary>
    SequenceRuns,
    /// <summary>Token type (segment) ids [T] i32 of encoders with a token type embedding table (BERT: 0 = first text, 1 = second).</summary>
    TokenTypes,
    /// <summary>
    /// Packed-sequence graphs (several independent sequences concatenated along the token axis, e.g. <see cref="WellKnown.EmbedGraph"/>):
    /// per-row table [T, 2] i32 of (first row, row count) of the sequence the row belongs to. Attention only looks within it.
    /// </summary>
    RowSpans,
    /// <summary>Packed-sequence graphs: per-sequence table [B, 2] i32 of (first row, row count); rows of one sequence are consecutive.</summary>
    SequenceSpans,
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
    /// <summary>Symbol of the run count of a multi-sequence graph (rows of <see cref="InputRole.SequenceRuns"/>).</summary>
    public const string RunSymbol = "U";

    /// <summary>
    /// Packed-sequence graph that turns a batch of sequences into one vector each (embedding models): inputs token ids [T],
    /// positions [T, 1] (each sequence starting at 0), optional token types [T], <see cref="InputRole.RowSpans"/> and
    /// <see cref="InputRole.SequenceSpans"/> [B, 2]; output <see cref="EmbeddingsOutput"/> [B, H] (pooled as the model's
    /// <see cref="EmbeddingPooling"/> property says and L2-normalized when <see cref="EmbeddingNormalize"/> is set) — or, for
    /// pooling "none", [T, H] (one vector per token).
    /// </summary>
    public const string EmbedGraph = "embed";
    /// <summary>
    /// Packed-sequence graph with the same inputs as <see cref="EmbedGraph"/> that scores each sequence with a head:
    /// output <see cref="LogitsOutput"/> [B, C] (cross-encoder classifier: C labels; decoder: the language model head over the
    /// sequence's last row, C = vocabulary).
    /// </summary>
    public const string ClassifyGraph = "classify";
    public const string EmbeddingsOutput = "embeddings";
    public const string LogitsOutput = "logits";
    /// <summary>Symbol of the sequence count of a packed-sequence graph (rows of <see cref="InputRole.SequenceSpans"/>).</summary>
    public const string SequenceSymbol = "B";

    public const string VocabSize = "vocab_size";
    public const string ContextLength = "context_length";
    public const string EmbeddingLength = "embedding_length";
    public const string PositionAxes = "position_axes";
    public const string EosTokenIds = "eos_token_ids";
    /// <summary>Pooling of <see cref="EmbedGraph"/>: "mean", "cls" (first row), "last", "none" (string).</summary>
    public const string EmbeddingPooling = "embedding_pooling";
    /// <summary>Whether <see cref="EmbedGraph"/> L2-normalizes its output (bool).</summary>
    public const string EmbeddingNormalize = "embedding_normalize";
    /// <summary>Whether attention of the packed-sequence graphs is causal (decoder-based embedders) (bool).</summary>
    public const string CausalAttention = "causal_attention";
    /// <summary>Label count of <see cref="ClassifyGraph"/> (int; the vocabulary size for decoders).</summary>
    public const string ClassifierLabels = "classifier_labels";
}
