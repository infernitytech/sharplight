namespace Sharplight.Import.Gguf;

/// <summary>
/// Reads GGUF metadata from the first bytes of a file, e.g. fetched with an HTTP range request before the file is downloaded. Only
/// the key/value pairs that lie entirely within those bytes are returned; the tensor table is read when the bytes cover it.
/// </summary>
public static unsafe class GgufHeaderProbe
{
    private const int HeaderBytes = 24;

    /// <summary>The most dimensions a tensor may have (as <see cref="GgufFile"/> accepts).</summary>
    private const uint MaxTensorDimensions = 8;

    /// <summary>The metadata key/values that the prefix contains completely, in file order (empty when it is not GGUF).</summary>
    /// <param name="prefix">The first bytes of a GGUF file.</param>
    public static IReadOnlyDictionary<string, object> ReadMetadata(ReadOnlySpan<byte> prefix)
    {
        var metadata = new Dictionary<string, object>(StringComparer.Ordinal);
        if (prefix.Length < HeaderBytes)
        {
            return metadata;
        }

        fixed (byte* data = prefix)
        {
            var reader = new GgufReader(data, prefix.Length);
            if (reader.ReadU32() != GgufFile.Magic || reader.ReadU32() is not (2 or 3))
            {
                return metadata;
            }

            reader.ReadU64();
            long kvCount = (long)Math.Min(reader.ReadU64(), int.MaxValue);
            for (long i = 0; i < kvCount; i++)
            {
                try
                {
                    string key = reader.ReadString();
                    var type = (GgufValueType)reader.ReadU32();
                    metadata[key] = reader.ReadValue(type);
                }
                catch (Exception error) when (error is InvalidDataException or OverflowException)
                {
                    // The prefix ends inside this pair (or the rest is malformed): keep what was read.
                    break;
                }
            }
        }

        return metadata;
    }

    /// <summary>
    /// The ggml type ids of the tensors in the tensor table, with the number of tensors of each, when the prefix holds the whole table
    /// (it follows the metadata, so the prefix must cover both); null when it ends earlier or is not GGUF. Of a split model, only the
    /// tensors of the shard the prefix belongs to are listed.
    /// </summary>
    /// <param name="prefix">The first bytes of a GGUF file.</param>
    /// <returns>Type id → tensor count; ids may be unknown to this version (see <see cref="GgufSupport.GetTensorType"/>).</returns>
    public static IReadOnlyDictionary<int, int>? ReadTensorTypes(ReadOnlySpan<byte> prefix)
        => ReadTensorTable(prefix) is { } table ? CountTypes(table) : null;

    /// <summary>The number of tensors of each ggml type id in a tensor table (see <see cref="ReadTensorTable"/>).</summary>
    /// <param name="tensors">The tensor table, of one shard or of all shards together.</param>
    public static IReadOnlyDictionary<int, int> CountTypes(IEnumerable<GgufTensorEntry> tensors)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        var types = new Dictionary<int, int>();
        foreach (var tensor in tensors)
        {
            types[tensor.GgmlType] = types.GetValueOrDefault(tensor.GgmlType) + 1;
        }

        return types;
    }

    /// <summary>
    /// Every entry of the tensor table (name, type id, dimensions), in file order, when the prefix holds the whole table (it follows the
    /// metadata, so the prefix must cover both); null when it ends earlier or is not GGUF. Of a split model, only the tensors of the
    /// shard the prefix belongs to are listed.
    /// </summary>
    /// <param name="prefix">The first bytes of a GGUF file.</param>
    public static IReadOnlyList<GgufTensorEntry>? ReadTensorTable(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < HeaderBytes)
        {
            return null;
        }

        fixed (byte* data = prefix)
        {
            var reader = new GgufReader(data, prefix.Length);
            try
            {
                if (reader.ReadU32() != GgufFile.Magic || reader.ReadU32() is not (2 or 3))
                {
                    return null;
                }

                ulong tensorCount = reader.ReadU64();
                ulong kvCount = reader.ReadU64();
                if (tensorCount > (ulong)prefix.Length || kvCount > (ulong)prefix.Length)
                {
                    return null;
                }

                for (ulong i = 0; i < kvCount; i++)
                {
                    reader.ReadString();
                    reader.ReadValue((GgufValueType)reader.ReadU32());
                }

                var tensors = new List<GgufTensorEntry>((int)tensorCount);
                for (ulong i = 0; i < tensorCount; i++)
                {
                    string name = reader.ReadString();
                    uint dimensions = reader.ReadU32();
                    if (dimensions > MaxTensorDimensions)
                    {
                        return null;
                    }

                    var ne = new long[dimensions];
                    for (uint d = 0; d < dimensions; d++)
                    {
                        ne[d] = (long)reader.ReadU64();
                    }

                    int type = (int)reader.ReadU32();
                    reader.ReadU64();
                    tensors.Add(new GgufTensorEntry(name, type, ne));
                }

                return tensors;
            }
            catch (Exception error) when (error is InvalidDataException or OverflowException)
            {
                // The prefix ends before the end of the tensor table (or the file is malformed).
                return null;
            }
        }
    }

    /// <summary>The model dimensions in the metadata the prefix contains (see <see cref="ReadMetadata"/>).</summary>
    /// <param name="prefix">The first bytes of a GGUF file.</param>
    public static GgufModelShape ReadShape(ReadOnlySpan<byte> prefix) => ShapeOf(ReadMetadata(prefix));

    /// <summary>The model dimensions in GGUF metadata.</summary>
    /// <param name="metadata">Key/values as <see cref="GgufFile.Metadata"/> or <see cref="ReadMetadata"/> return them.</param>
    public static GgufModelShape ShapeOf(IReadOnlyDictionary<string, object> metadata)
    {
        ArgumentNullException.ThrowIfNull(metadata);
        var arch = metadata.TryGetValue("general.architecture", out var name) ? name as string : null;
        int? vocabulary = metadata.TryGetValue("tokenizer.ggml.tokens", out var tokens) && tokens is Array list ? list.Length : null;
        if (arch is null)
        {
            return new GgufModelShape(null, null, null, null, null, null, null, null, vocabulary);
        }

        int? Get(string key) => Largest(metadata, $"{arch}.{key}");
        return new GgufModelShape(arch, Get("block_count"), Get("attention.head_count"), Get("attention.head_count_kv") ?? Get("attention.head_count"),
            Get("embedding_length"), Get("attention.key_length"), Get("attention.value_length"), Get("context_length"), vocabulary,
            Get("attention.kv_lora_rank"), Get("rope.dimension_count"));
    }

    /// <summary>A numeric value, or the largest element of a per-layer array.</summary>
    private static int? Largest(IReadOnlyDictionary<string, object> metadata, string key)
    {
        if (!metadata.TryGetValue(key, out var raw))
        {
            return null;
        }

        if (raw is Array values)
        {
            long? largest = null;
            foreach (var value in values)
            {
                if (value is not null && GgufConvert.TryConvert(value, out long converted) && (largest is null || converted > largest))
                {
                    largest = converted;
                }
            }

            return largest is { } found && found <= int.MaxValue ? (int)found : null;
        }

        return GgufConvert.TryConvert(raw, out long single) && single is >= 0 and <= int.MaxValue ? (int)single : null;
    }
}
