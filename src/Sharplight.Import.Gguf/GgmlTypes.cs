using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.Gguf;

/// <summary>Mapping between ggml_type ids (as stored in GGUF) and <see cref="DataType"/>.</summary>
internal static class GgmlTypes
{
    /// <summary>PrismML fork id (llama.cpp "prism" branch) of PQ2_0, a group-128 Q2_0.</summary>
    internal const int PrismPQ2_0 = 142;

    /// <summary>PrismML fork id (llama.cpp "prism" branch) of PTQ1_0, a group-128 dense ternary type.</summary>
    internal const int PrismPTQ1_0 = 143;

    /// <summary>
    /// The trellis types of agentionai's llama.cpp fork (TQ2_T, TQK6, TQK7, TQP5, TQP6, TQP7: ids 144-149, the Gyro TQ*_0
    /// releases), in id order.
    /// </summary>
    private static readonly DataType[] s_trellisById =
    [
        DataType.TrellisV4K8_G128, DataType.TrellisV4K6_G128, DataType.TrellisV4K7_G128,
        DataType.TrellisV2K5_G128, DataType.TrellisV2K6_G128, DataType.TrellisV2K7_G128,
    ];

    /// <summary>The ggml id of the first trellis type of agentionai's fork (TQ2_T).</summary>
    internal const int FirstTrellis = 144;

    /// <summary>
    /// ggml id of I2_S, the ternary type of Microsoft's BitNet fork of llama.cpp: 2-bit codes in 128-value groups followed, after the
    /// codes of the whole tensor, by one f32 scale (see <see cref="GgufI2sSource"/>). Read as <see cref="DataType.Q2S_G128"/>.
    /// </summary>
    internal const int I2S = 36;

    /// <summary>The interleaved Q4_0 types of older llama.cpp releases (Q4_0_4_4, Q4_0_4_8, Q4_0_8_8), read as Q4_0.</summary>
    internal const int Q4_0_4_4 = 31;

    /// <inheritdoc cref="Q4_0_4_4"/>
    internal const int Q4_0_4_8 = 32;

    /// <inheritdoc cref="Q4_0_4_4"/>
    internal const int Q4_0_8_8 = 33;

    // Types without a static in DataType are registered here (lowercase ggml names).
    internal static readonly DataType Q8_1 = DataType.Register("q8_1", DataTypeKind.Quantized, 32, 36);
    internal static readonly DataType F8_E4M3 = DataType.Register("f8_e4m3", DataTypeKind.Float, 1, 1);

    private static readonly DataType?[] s_byId = Build();
    private static readonly Dictionary<DataType, int> s_toId = BuildReverse();

    /// <summary>Builds the id table; ids that llama.cpp removed or that are not supported stay null.</summary>
    private static DataType?[] Build()
    {
        var types = new DataType?[44];
        types[0] = DataType.F32;
        types[1] = DataType.F16;
        types[2] = DataType.Q4_0;
        types[3] = DataType.Q4_1;
        types[6] = DataType.Q5_0;
        types[7] = DataType.Q5_1;
        types[8] = DataType.Q8_0;
        types[9] = Q8_1;
        types[10] = DataType.Q2_K;
        types[11] = DataType.Q3_K;
        types[12] = DataType.Q4_K;
        types[13] = DataType.Q5_K;
        types[14] = DataType.Q6_K;
        types[15] = DataType.Q8_K;
        types[16] = DataType.IQ2_XXS;
        types[17] = DataType.IQ2_XS;
        types[18] = DataType.IQ3_XXS;
        types[19] = DataType.IQ1_S;
        types[20] = DataType.IQ4_NL;
        types[21] = DataType.IQ3_S;
        types[22] = DataType.IQ2_S;
        types[23] = DataType.IQ4_XS;
        types[24] = DataType.I8;
        types[25] = DataType.I16;
        types[26] = DataType.I32;
        types[27] = DataType.I64;
        types[28] = DataType.F64;
        types[29] = DataType.IQ1_M;
        types[30] = DataType.BF16;
        types[34] = DataType.TQ1_0;
        types[35] = DataType.TQ2_0;
        types[I2S] = DataType.Q2S_G128;
        types[39] = DataType.MXFP4;
        types[40] = DataType.NVFP4;
        types[41] = DataType.Q1_0;
        types[42] = DataType.Q2_0;
        types[43] = F8_E4M3;
        return types;
    }

    private static Dictionary<DataType, int> BuildReverse()
    {
        var reverse = new Dictionary<DataType, int>();
        for (int id = 0; id < s_byId.Length; id++)
        {
            if (s_byId[id] is { } type && id != I2S)
            {
                reverse[type] = id;
            }
        }

        return reverse;
    }

    /// <summary>Returns the data type for a ggml type id, or null if unknown/removed.</summary>
    public static DataType? TryGetDataType(int ggmlType)
    {
        if (TryGetQ4_0Interleave(ggmlType, out _, out _))
        {
            return DataType.Q4_0;
        }

        if (GgmlNarrowIq1.Is(ggmlType))
        {
            return DataType.IQ1_S;
        }

        if ((uint)ggmlType < s_byId.Length)
        {
            return s_byId[ggmlType];
        }

        if (ggmlType >= FirstTrellis && ggmlType < FirstTrellis + s_trellisById.Length)
        {
            return s_trellisById[ggmlType - FirstTrellis];
        }

        return ggmlType switch
        {
            PrismPQ2_0 => DataType.Q2_0_G128,
            PrismPTQ1_0 => DataType.Tern1_G128,
            _ => null,
        };
    }

    /// <summary>
    /// The layout of an interleaved Q4_0 type id: rows interleaved per group and the bytes of a block's nibble array kept together;
    /// false for every other id.
    /// </summary>
    public static bool TryGetQ4_0Interleave(int ggmlType, out int rows, out int chunk)
    {
        (rows, chunk) = ggmlType switch
        {
            Q4_0_4_4 => (4, 4),
            Q4_0_4_8 => (4, 8),
            Q4_0_8_8 => (8, 8),
            _ => (0, 0),
        };
        return rows != 0;
    }

    /// <summary>Returns the data type for a ggml type id.</summary>
    /// <exception cref="UnsupportedWeightFormatException">The id is unknown or removed.</exception>
    public static DataType GetDataType(int ggmlType) => TryGetDataType(ggmlType)
        ?? throw new UnsupportedWeightFormatException($"ggml type id {ggmlType}", $"Unsupported ggml tensor type id {ggmlType}.");

    /// <summary>Returns the ggml type id for a data type.</summary>
    /// <exception cref="NotSupportedException">The type has no ggml id.</exception>
    public static int GetGgmlType(DataType type) => TryGetGgmlType(type, out var id)
        ? id : throw new NotSupportedException($"Data type '{type}' has no ggml type id.");

    /// <summary>Returns the ggml type id for a data type, or false when it has none.</summary>
    public static bool TryGetGgmlType(DataType type, out int id)
    {
        if (type == DataType.Q2_0_G128)
        {
            id = PrismPQ2_0;
            return true;
        }

        if (type == DataType.Tern1_G128)
        {
            id = PrismPTQ1_0;
            return true;
        }

        int trellis = Array.IndexOf(s_trellisById, type);
        if (trellis >= 0)
        {
            id = FirstTrellis + trellis;
            return true;
        }

        return s_toId.TryGetValue(type, out id);
    }
}
