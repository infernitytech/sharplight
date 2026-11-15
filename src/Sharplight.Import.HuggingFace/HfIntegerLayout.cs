namespace Sharplight.Import.HuggingFace;

/// <summary>
/// One packed integer weight of a checkpoint (GPTQ, AWQ, compressed-tensors) described independently of its packing: the logical
/// [rows, columns] matrix of unsigned codes with a scale and a zero point per group of columns, value = (code - zero) * scale.
/// </summary>
internal sealed class HfIntegerLayout
{
    /// <summary>The canonical-to-be name of the weight (<c>X.weight</c>).</summary>
    public required string Name { get; init; }

    /// <summary>A tensor of the checkpoint the weight comes from (the converted tensor reuses its file entry).</summary>
    public required SafetensorsTensor Origin { get; init; }

    /// <summary>Bits per code (2 to 8).</summary>
    public required int Bits { get; init; }

    /// <summary>Output rows N.</summary>
    public required long Rows { get; init; }

    /// <summary>Input columns K.</summary>
    public required long Columns { get; init; }

    /// <summary>Columns per group (K for one group per row).</summary>
    public required long GroupSize { get; init; }

    /// <summary>
    /// The group of each column when the checkpoint stores one (GPTQ <c>g_idx</c>, act-order), else null (column k is in group
    /// k / <see cref="GroupSize"/>).
    /// </summary>
    public int[]? GroupIndex { get; init; }

    /// <summary>
    /// Decodes row <c>n</c> (the first argument): its K unsigned codes (in column order) into the first span, and the scale of each
    /// group and its zero point (an exact value, integer for every integer format) into the other two.
    /// </summary>
    public required HfIntegerRowDecoder DecodeRow { get; init; }

    /// <summary>
    /// Whether every scale is exactly an f16 value (always for f16 scales; for BF16 / F32 scales the importer checks them all).
    /// When not, the weight cannot keep the f16 scales of the integer block types and is decoded to BF16.
    /// </summary>
    public required bool ScalesFitHalf { get; init; }

    /// <summary>
    /// Whether the third value <see cref="DecodeRow"/> produces per group is an additive bias instead of a zero point, value =
    /// scale * code + bias (AWQ <c>gemv_fast</c>, which stores -scale * zero in f16). Such weights keep their codes as
    /// <see cref="Sharplight.Models.DataType.Q4_1"/> / <see cref="Sharplight.Models.DataType.Q5_1"/> blocks (<see cref="HfAffineWeights"/>).
    /// </summary>
    public bool Affine { get; init; }

    /// <summary>Groups per row.</summary>
    public long Groups => (Columns + GroupSize - 1) / GroupSize;
}
