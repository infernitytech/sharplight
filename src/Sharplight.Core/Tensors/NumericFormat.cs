namespace Sharplight.Tensors;

/// <summary>
/// The value set of a numeric element format, independent of how tensors of it are stored. Used to round (floats) or wrap/clamp
/// (integers) values to what a narrower format can represent while they travel in a wider storage type (see <see cref="NumericFormats"/>).
/// </summary>
public enum NumericFormat
{
    /// <summary>IEEE half precision.</summary>
    F16,

    /// <summary>bfloat16 (8 exponent bits, 7 mantissa bits).</summary>
    BF16,

    /// <summary>OCP FP8 E4M3FN: bias 7, no infinity, NaN = S.1111.111, largest finite 448.</summary>
    F8E4M3FN,

    /// <summary>FP8 E4M3FNUZ: bias 8, no infinity and no negative zero, NaN = 0x80, largest finite 240.</summary>
    F8E4M3FNUZ,

    /// <summary>OCP FP8 E5M2: bias 15 with IEEE infinities and NaNs, largest finite 57344.</summary>
    F8E5M2,

    /// <summary>FP8 E5M2FNUZ: bias 16, no infinity and no negative zero, NaN = 0x80, largest finite 57344.</summary>
    F8E5M2FNUZ,

    /// <summary>FP4 E2M1: bias 1, no infinity or NaN, values ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}.</summary>
    F4E2M1,

    /// <summary>E8M0 power-of-two scale: 2^(code - 127), code 255 = NaN, no sign and no zero.</summary>
    F8E8M0,

    /// <summary>Signed 2-bit integer (-2..1).</summary>
    I2,

    /// <summary>Unsigned 2-bit integer (0..3).</summary>
    U2,

    /// <summary>Signed 4-bit integer (-8..7).</summary>
    I4,

    /// <summary>Unsigned 4-bit integer (0..15).</summary>
    U4,

    /// <summary>Signed 8-bit integer.</summary>
    I8,

    /// <summary>Unsigned 8-bit integer.</summary>
    U8,

    /// <summary>Signed 16-bit integer.</summary>
    I16,

    /// <summary>Unsigned 16-bit integer.</summary>
    U16,

    /// <summary>Signed 32-bit integer.</summary>
    I32,

    /// <summary>Unsigned 32-bit integer.</summary>
    U32,
}
