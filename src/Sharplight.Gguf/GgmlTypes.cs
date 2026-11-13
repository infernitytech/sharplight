using Sharplight.Tensors;

namespace Sharplight.Gguf;

/// <summary>Mapping between ggml_type ids (as stored in GGUF) and <see cref="DataType"/>.</summary>
public static class GgmlTypes
{
    // Types without a static in DataType are registered here (lowercase ggml names).
    public static readonly DataType Q1_0 = DataType.Register("q1_0", DataTypeKind.Quantized, 128, 18);
    public static readonly DataType Q2_0 = DataType.Register("q2_0", DataTypeKind.Quantized, 64, 18);
    public static readonly DataType Q8_1 = DataType.Register("q8_1", DataTypeKind.Quantized, 32, 36);
    public static readonly DataType IQ2_XXS = DataType.Register("iq2_xxs", DataTypeKind.Quantized, 256, 66);
    public static readonly DataType IQ2_XS = DataType.Register("iq2_xs", DataTypeKind.Quantized, 256, 74);
    public static readonly DataType IQ2_S = DataType.Register("iq2_s", DataTypeKind.Quantized, 256, 82);
    public static readonly DataType IQ3_XXS = DataType.Register("iq3_xxs", DataTypeKind.Quantized, 256, 98);
    public static readonly DataType IQ3_S = DataType.Register("iq3_s", DataTypeKind.Quantized, 256, 110);
    public static readonly DataType IQ1_S = DataType.Register("iq1_s", DataTypeKind.Quantized, 256, 50);
    public static readonly DataType IQ1_M = DataType.Register("iq1_m", DataTypeKind.Quantized, 256, 56);
    public static readonly DataType TQ1_0 = DataType.Register("tq1_0", DataTypeKind.Quantized, 256, 54);
    public static readonly DataType TQ2_0 = DataType.Register("tq2_0", DataTypeKind.Quantized, 256, 66);
    public static readonly DataType NVFP4 = DataType.Register("nvfp4", DataTypeKind.Quantized, 64, 36);
    public static readonly DataType F8_E4M3 = DataType.Register("f8_e4m3", DataTypeKind.Float, 1, 1);

    static readonly DataType?[] s_byId = Build();
    static readonly Dictionary<DataType, int> s_toId = BuildReverse();

    static DataType?[] Build()
    {
        var t = new DataType?[44];
        t[0] = DataType.F32; t[1] = DataType.F16; t[2] = DataType.Q4_0; t[3] = DataType.Q4_1;
        t[6] = DataType.Q5_0; t[7] = DataType.Q5_1; t[8] = DataType.Q8_0; t[9] = Q8_1;
        t[10] = DataType.Q2_K; t[11] = DataType.Q3_K; t[12] = DataType.Q4_K; t[13] = DataType.Q5_K;
        t[14] = DataType.Q6_K; t[15] = DataType.Q8_K;
        t[16] = IQ2_XXS; t[17] = IQ2_XS; t[18] = IQ3_XXS; t[19] = IQ1_S; t[20] = DataType.IQ4_NL;
        t[21] = IQ3_S; t[22] = IQ2_S; t[23] = DataType.IQ4_XS;
        t[24] = DataType.I8; t[25] = DataType.I16; t[26] = DataType.I32; t[27] = DataType.I64;
        t[28] = DataType.F64; t[29] = IQ1_M; t[30] = DataType.BF16;
        t[34] = TQ1_0; t[35] = TQ2_0; t[39] = DataType.MXFP4; t[40] = NVFP4;
        t[41] = Q1_0; t[42] = Q2_0; t[43] = F8_E4M3;
        return t;
    }

    static Dictionary<DataType, int> BuildReverse()
    {
        var d = new Dictionary<DataType, int>();
        for (int i = 0; i < s_byId.Length; i++) if (s_byId[i] is { } dt) d[dt] = i;
        return d;
    }

    /// <summary>Returns the data type for a ggml type id, or null if unknown/removed.</summary>
    public static DataType? TryGetDataType(int ggmlType) => (uint)ggmlType < s_byId.Length ? s_byId[ggmlType] : null;

    public static DataType GetDataType(int ggmlType) => TryGetDataType(ggmlType)
        ?? throw new NotSupportedException($"Unsupported ggml tensor type id {ggmlType}.");

    public static int GetGgmlType(DataType type) => s_toId.TryGetValue(type, out var id)
        ? id : throw new NotSupportedException($"Data type '{type}' has no ggml type id.");

    public static bool TryGetGgmlType(DataType type, out int id) => s_toId.TryGetValue(type, out id);
}
