using Sharplight.Tensors;

namespace Sharplight.Gguf;

/// <summary>Mapping between ggml_type ids (as stored in GGUF) and <see cref="DataType"/>.</summary>
internal static class GgmlTypes
{
    /// <summary>PrismML fork id (llama.cpp "prism" branch) of PQ2_0, a group-128 Q2_0.</summary>
    internal const int PrismPQ2_0 = 142;

    /// <summary>PrismML fork id (llama.cpp "prism" branch) of PTQ1_0, a group-128 dense ternary type.</summary>
    internal const int PrismPTQ1_0 = 143;

    // Types without a static in DataType are registered here (lowercase ggml names).
    internal static readonly DataType Q8_1 = DataType.Register("q8_1", DataTypeKind.Quantized, 32, 36);
    internal static readonly DataType IQ2_XXS = DataType.Register("iq2_xxs", DataTypeKind.Quantized, 256, 66);
    internal static readonly DataType IQ2_XS = DataType.Register("iq2_xs", DataTypeKind.Quantized, 256, 74);
    internal static readonly DataType IQ2_S = DataType.Register("iq2_s", DataTypeKind.Quantized, 256, 82);
    internal static readonly DataType IQ3_XXS = DataType.Register("iq3_xxs", DataTypeKind.Quantized, 256, 98);
    internal static readonly DataType IQ3_S = DataType.Register("iq3_s", DataTypeKind.Quantized, 256, 110);
    internal static readonly DataType IQ1_S = DataType.Register("iq1_s", DataTypeKind.Quantized, 256, 50);
    internal static readonly DataType IQ1_M = DataType.Register("iq1_m", DataTypeKind.Quantized, 256, 56);
    internal static readonly DataType TQ1_0 = DataType.Register("tq1_0", DataTypeKind.Quantized, 256, 54);
    internal static readonly DataType TQ2_0 = DataType.Register("tq2_0", DataTypeKind.Quantized, 256, 66);
    internal static readonly DataType NVFP4 = DataType.Register("nvfp4", DataTypeKind.Quantized, 64, 36);
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
        types[16] = IQ2_XXS;
        types[17] = IQ2_XS;
        types[18] = IQ3_XXS;
        types[19] = IQ1_S;
        types[20] = DataType.IQ4_NL;
        types[21] = IQ3_S;
        types[22] = IQ2_S;
        types[23] = DataType.IQ4_XS;
        types[24] = DataType.I8;
        types[25] = DataType.I16;
        types[26] = DataType.I32;
        types[27] = DataType.I64;
        types[28] = DataType.F64;
        types[29] = IQ1_M;
        types[30] = DataType.BF16;
        types[34] = TQ1_0;
        types[35] = TQ2_0;
        types[39] = DataType.MXFP4;
        types[40] = NVFP4;
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
            if (s_byId[id] is { } type)
            {
                reverse[type] = id;
            }
        }

        return reverse;
    }

    /// <summary>Returns the data type for a ggml type id, or null if unknown/removed.</summary>
    public static DataType? TryGetDataType(int ggmlType)
    {
        if ((uint)ggmlType < s_byId.Length)
        {
            return s_byId[ggmlType];
        }

        return ggmlType switch
        {
            PrismPQ2_0 => DataType.Q2_0_G128,
            PrismPTQ1_0 => DataType.Tern1_G128,
            _ => null,
        };
    }

    /// <summary>Returns the data type for a ggml type id.</summary>
    /// <exception cref="NotSupportedException">The id is unknown or removed.</exception>
    public static DataType GetDataType(int ggmlType) => TryGetDataType(ggmlType)
        ?? throw new NotSupportedException($"Unsupported ggml tensor type id {ggmlType}.");

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

        return s_toId.TryGetValue(type, out id);
    }
}
