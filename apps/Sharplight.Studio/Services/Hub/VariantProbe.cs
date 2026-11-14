using Sharplight.Gguf;
using Sharplight.Onnx;
using Sharplight.Studio.Services.Compatibility;

namespace Sharplight.Studio.Services.Hub;

/// <summary>
/// Reads what a model file states about itself from ranges of it, before (or without) downloading it: the GGUF header and tensor
/// table from the start of the file, an ONNX model's opset imports from the bytes around its graph.
/// </summary>
internal static class VariantProbe
{
    /// <summary>Bytes read from the start of a GGUF file, in steps: the first usually holds the dimensions, a later one the tensor table.</summary>
    private static readonly int[] s_ggufPrefixes = [1 << 20, 8 << 20, 32 << 20, MaxGgufPrefixBytes];

    /// <summary>The most bytes read from the start of a GGUF file (or of each shard) for its tensor table.</summary>
    public const int MaxGgufPrefixBytes = 64 << 20;

    /// <summary>Bytes read per request while looking for an ONNX model's opset imports.</summary>
    private const int OnnxChunkBytes = 64 << 10;

    /// <summary>Requests made at most for the opset imports.</summary>
    private const int OnnxReads = 8;

    /// <summary>
    /// The dimensions and compatibility of a GGUF file from its first bytes: more are read only while the dimensions (or, with
    /// <paramref name="readWeightFormats"/>, the tensor table) lie beyond them, up to <see cref="MaxGgufPrefixBytes"/>. With the weight
    /// formats the whole tensor table of every shard is read and checked against the importer (<see cref="CompatibilityChecker.FromGguf"/>);
    /// when any of it lies beyond what is read, the verdict is unknown rather than supported.
    /// </summary>
    /// <param name="read">Reads up to <c>count</c> bytes of the file starting at <c>offset</c> (fewer at its end).</param>
    /// <param name="readWeightFormats">Also read the tensor table for the tensor types and names.</param>
    /// <param name="cancellationToken">Cancels the reads.</param>
    /// <param name="otherShards">Readers of the second and later shards of a split model, in order; null or empty for a single file.</param>
    public static async Task<ModelProbe> ProbeGgufAsync(Func<long, int, CancellationToken, Task<byte[]>> read, bool readWeightFormats,
        CancellationToken cancellationToken, IReadOnlyList<Func<long, int, CancellationToken, Task<byte[]>>>? otherShards = null)
    {
        var first = await ReadHeaderAsync(read, readWeightFormats, wantShape: true, cancellationToken).ConfigureAwait(false);
        var metadata = first.Metadata;
        var shape = ModelShapeReader.FromGguf(GgufHeaderProbe.ShapeOf(metadata));
        var architecture = CompatibilityChecker.FromGguf(metadata, null) with { WeightsChecked = false };
        if (!readWeightFormats || architecture.State is CompatibilityState.UnsupportedArchitecture or CompatibilityState.Projector or CompatibilityState.Unknown)
        {
            return new ModelProbe(shape, architecture, architecture);
        }

        if (first.Tensors is not { } tensors)
        {
            return new ModelProbe(shape, BeyondPrefix(architecture, "its tensor table"), architecture);
        }

        var all = new List<GgufTensorEntry>(tensors);
        var shards = otherShards ?? [];
        for (int i = 0; i < shards.Count; i++)
        {
            var shard = await ReadHeaderAsync(shards[i], readWeightFormats: true, wantShape: false, cancellationToken).ConfigureAwait(false);
            if (shard.Tensors is not { } shardTensors)
            {
                return new ModelProbe(shape, BeyondPrefix(architecture, $"the tensor table of shard {i + 2}"), architecture);
            }

            all.AddRange(shardTensors);
        }

        return new ModelProbe(shape, CompatibilityChecker.FromGguf(metadata, all), architecture);
    }

    /// <summary>The metadata and, when asked for, the tensor table of one GGUF file, read in growing steps.</summary>
    private static async Task<(IReadOnlyDictionary<string, object> Metadata, IReadOnlyList<GgufTensorEntry>? Tensors)> ReadHeaderAsync(
        Func<long, int, CancellationToken, Task<byte[]>> read, bool readWeightFormats, bool wantShape, CancellationToken cancellationToken)
    {
        byte[] prefix = [];
        IReadOnlyDictionary<string, object> metadata = new Dictionary<string, object>();
        IReadOnlyList<GgufTensorEntry>? tensors = null;
        foreach (int target in s_ggufPrefixes)
        {
            var more = await read(prefix.Length, target - prefix.Length, cancellationToken).ConfigureAwait(false);
            prefix = [.. prefix, .. more];
            metadata = GgufHeaderProbe.ReadMetadata(prefix);
            tensors = readWeightFormats ? GgufHeaderProbe.ReadTensorTable(prefix) : null;
            bool settled = CompatibilityChecker.FromGguf(metadata, null).State is CompatibilityState.UnsupportedArchitecture or CompatibilityState.Projector
                or CompatibilityState.DraftModel;
            bool hasShape = !wantShape || ModelShapeReader.FromGguf(GgufHeaderProbe.ShapeOf(metadata)) is not null;
            if (prefix.Length < target || (settled && wantShape) || (hasShape && (!readWeightFormats || tensors is not null)))
            {
                break;
            }
        }

        return (metadata, tensors);
    }

    /// <summary>The verdict when part of a tensor table lies beyond the bytes read: unknown, as nothing proves that it imports.</summary>
    private static ModelCompatibility BeyondPrefix(ModelCompatibility architecture, string what)
        => ModelCompatibility.Unknown($"{architecture.Reason} Its weights could not be checked: {what} lies beyond the first "
            + $"{MaxGgufPrefixBytes >> 20} MiB read before downloading, so it is checked when the model is imported.");

    /// <summary>The compatibility of an ONNX model: its opset imports, found by skipping the graph (operators are checked at import).</summary>
    /// <param name="read">Reads up to <c>count</c> bytes of the file starting at <c>offset</c> (fewer at its end).</param>
    /// <param name="fileLength">The size of the file.</param>
    /// <param name="cancellationToken">Cancels the reads.</param>
    public static async Task<ModelCompatibility> ProbeOnnxAsync(Func<long, int, CancellationToken, Task<byte[]>> read, long fileLength, CancellationToken cancellationToken)
    {
        if (fileLength <= 0)
        {
            return CompatibilityChecker.FromOnnx(null);
        }

        var scanner = new OnnxOpsetScanner(fileLength);
        for (int i = 0; i < OnnxReads && !scanner.IsComplete && !scanner.IsMalformed; i++)
        {
            long before = scanner.Position;
            var bytes = await read(scanner.Position, (int)Math.Min(OnnxChunkBytes, fileLength - scanner.Position), cancellationToken).ConfigureAwait(false);
            if (bytes.Length == 0)
            {
                break;
            }

            scanner.Feed(bytes);
            if (scanner.Position == before)
            {
                break;
            }
        }

        return CompatibilityChecker.FromOnnx(scanner.IsComplete || scanner.Opsets.Count > 0 ? scanner.Opsets : null);
    }
}
