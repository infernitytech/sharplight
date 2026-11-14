namespace Sharplight.Tensors;

/// <summary>
/// Conversions to and from the value sets of <see cref="NumericFormat"/>s: rounding a value to what a narrow format can hold
/// (<see cref="Round"/>), and encoding/decoding the format's bit patterns (<see cref="Encode"/>, <see cref="Decode"/>).
/// </summary>
/// <remarks>
/// Floats round to nearest, ties to even (<see cref="NumericFormat.F8E8M0"/> uses an <see cref="ExponentRounding"/>). Out-of-range
/// values of the FP8 formats saturate to the largest finite magnitude when <c>saturate</c> is set; otherwise they become infinity
/// (E5M2) or NaN (the formats without infinity). FP4 E2M1 always saturates and maps NaN to -0. Integers truncate toward zero and then
/// wrap modulo 2^bits, or clamp to the format's range when <c>saturate</c> is set; NaN converts to 0.
/// </remarks>
public static class NumericFormats
{
    private enum Specials
    {
        Ieee,
        FiniteOrNaN,
        Fnuz,
        Finite,
    }

    private readonly record struct MiniFloat(int ExponentBits, int MantissaBits, int Bias, Specials Specials)
    {
        public int MaxExponentField => (1 << ExponentBits) - 1;

        public int SignBit => 1 << (ExponentBits + MantissaBits);

        public double Max => Specials switch
        {
            Specials.Ieee => (2 - Math.ScaleB(1.0, -MantissaBits)) * Math.ScaleB(1.0, MaxExponentField - 1 - Bias),
            Specials.FiniteOrNaN => (2 - Math.ScaleB(1.0, 1 - MantissaBits)) * Math.ScaleB(1.0, MaxExponentField - Bias),
            _ => (2 - Math.ScaleB(1.0, -MantissaBits)) * Math.ScaleB(1.0, MaxExponentField - Bias),
        };
    }

    private static MiniFloat? Mini(NumericFormat format) => format switch
    {
        NumericFormat.F8E4M3FN => new MiniFloat(4, 3, 7, Specials.FiniteOrNaN),
        NumericFormat.F8E4M3FNUZ => new MiniFloat(4, 3, 8, Specials.Fnuz),
        NumericFormat.F8E5M2 => new MiniFloat(5, 2, 15, Specials.Ieee),
        NumericFormat.F8E5M2FNUZ => new MiniFloat(5, 2, 16, Specials.Fnuz),
        NumericFormat.F4E2M1 => new MiniFloat(2, 1, 1, Specials.Finite),
        _ => null,
    };

    /// <summary>Lower-case name of a format (as used in IR op attributes), for example <c>f8e4m3fn</c> or <c>u4</c>.</summary>
    /// <param name="format">The format.</param>
    public static string Name(NumericFormat format) => format.ToString().ToLowerInvariant();

    /// <summary>Parses a name produced by <see cref="Name"/> (case-insensitive).</summary>
    /// <param name="name">Format name.</param>
    /// <exception cref="ArgumentException">Unknown name.</exception>
    public static NumericFormat Parse(string name)
        => Enum.TryParse<NumericFormat>(name, ignoreCase: true, out var format) && Enum.IsDefined(format)
            ? format
            : throw new ArgumentException($"Unknown numeric format '{name}'.", nameof(name));

    /// <summary>Whether the format is a floating-point format.</summary>
    /// <param name="format">The format.</param>
    public static bool IsFloat(NumericFormat format) => format <= NumericFormat.F8E8M0;

    /// <summary>Width of one value in bits.</summary>
    /// <param name="format">The format.</param>
    public static int Bits(NumericFormat format) => format switch
    {
        NumericFormat.I2 or NumericFormat.U2 => 2,
        NumericFormat.I4 or NumericFormat.U4 or NumericFormat.F4E2M1 => 4,
        NumericFormat.F16 or NumericFormat.BF16 or NumericFormat.I16 or NumericFormat.U16 => 16,
        NumericFormat.I32 or NumericFormat.U32 => 32,
        _ => 8,
    };

    /// <summary>Range of an integer format.</summary>
    /// <param name="format">An integer format.</param>
    /// <exception cref="ArgumentException">The format is a float format.</exception>
    public static (long Min, long Max) IntegerRange(NumericFormat format)
    {
        if (IsFloat(format))
        {
            throw new ArgumentException($"{format} is not an integer format.", nameof(format));
        }

        int bits = Bits(format);
        bool signed = format is NumericFormat.I2 or NumericFormat.I4 or NumericFormat.I8 or NumericFormat.I16 or NumericFormat.I32;
        return signed ? (-(1L << (bits - 1)), (1L << (bits - 1)) - 1) : (0, (1L << bits) - 1);
    }

    /// <summary>Largest finite magnitude of a float format.</summary>
    /// <param name="format">A float format.</param>
    public static double MaxFinite(NumericFormat format) => format switch
    {
        NumericFormat.F16 => (double)Half.MaxValue,
        NumericFormat.BF16 => BitConverter.UInt32BitsToSingle(0x7F7F0000),
        NumericFormat.F8E8M0 => Math.ScaleB(1.0, 127),
        _ when Mini(format) is { } mini => mini.Max,
        _ => throw new ArgumentException($"{format} is not a float format.", nameof(format)),
    };

    /// <summary>The value <paramref name="x"/> converts to in <paramref name="format"/>.</summary>
    /// <param name="format">Target format.</param>
    /// <param name="x">Value (an f32 value for float formats, exact results assume that).</param>
    /// <param name="saturate">Clamp out-of-range values (FP8 and integer formats) instead of overflowing.</param>
    /// <param name="rounding">Rounding of <see cref="NumericFormat.F8E8M0"/>.</param>
    public static double Round(NumericFormat format, double x, bool saturate = true, ExponentRounding rounding = ExponentRounding.Up)
    {
        switch (format)
        {
            case NumericFormat.F16:
                return (double)(Half)x;
            case NumericFormat.BF16:
                return RoundBFloat16((float)x);
            case NumericFormat.F8E8M0:
                return Decode(format, EncodeE8M0((float)x, saturate, rounding));
        }

        if (Mini(format) is { } mini)
        {
            return RoundMini(mini, x, saturate);
        }

        var (min, max) = IntegerRange(format);
        if (double.IsNaN(x))
        {
            return 0;
        }

        double t = Math.Truncate(x);
        if (saturate)
        {
            return Math.Clamp(t, min, max);
        }

        long v = t >= long.MaxValue ? long.MaxValue : t <= long.MinValue ? long.MinValue : (long)t;
        return Wrap(v, Bits(format), min < 0);
    }

    /// <summary>Wraps an integer to <paramref name="bits"/> bits (two's complement when <paramref name="signed"/>).</summary>
    /// <param name="value">The value.</param>
    /// <param name="bits">Width, 1 to 63.</param>
    /// <param name="signed">Interpret the low bits as signed.</param>
    public static long Wrap(long value, int bits, bool signed)
    {
        long mask = (1L << bits) - 1;
        long low = value & mask;
        return signed && (low >> (bits - 1)) != 0 ? low - (1L << bits) : low;
    }

    /// <summary>The bit pattern (low <see cref="Bits"/> bits) that encodes <paramref name="x"/> in <paramref name="format"/>.</summary>
    /// <param name="format">Target format.</param>
    /// <param name="x">Value.</param>
    /// <param name="saturate">As in <see cref="Round"/>.</param>
    /// <param name="rounding">As in <see cref="Round"/>.</param>
    public static int Encode(NumericFormat format, double x, bool saturate = true, ExponentRounding rounding = ExponentRounding.Up)
    {
        switch (format)
        {
            case NumericFormat.F16:
                return BitConverter.HalfToUInt16Bits((Half)x);
            case NumericFormat.BF16:
                return (int)(BitConverter.SingleToUInt32Bits((float)RoundBFloat16((float)x)) >> 16);
            case NumericFormat.F8E8M0:
                return EncodeE8M0((float)x, saturate, rounding);
        }

        if (Mini(format) is { } mini)
        {
            return EncodeMini(mini, RoundMini(mini, x, saturate));
        }

        long v = (long)Round(format, x, saturate);
        return (int)(v & ((1L << Bits(format)) - 1));
    }

    /// <summary>The value of the bit pattern <paramref name="code"/> (low <see cref="Bits"/> bits) of <paramref name="format"/>.</summary>
    /// <param name="format">The format.</param>
    /// <param name="code">Bit pattern.</param>
    public static double Decode(NumericFormat format, int code)
    {
        switch (format)
        {
            case NumericFormat.F16:
                return (double)BitConverter.UInt16BitsToHalf((ushort)code);
            case NumericFormat.BF16:
                return BitConverter.UInt32BitsToSingle((uint)(code & 0xFFFF) << 16);
            case NumericFormat.F8E8M0:
                return (code & 0xFF) == 0xFF ? double.NaN : Math.ScaleB(1.0, (code & 0xFF) - 127);
        }

        if (Mini(format) is { } mini)
        {
            return DecodeMini(mini, code);
        }

        int bits = Bits(format);
        long raw = bits == 32 ? (uint)code : code & ((1L << bits) - 1);
        return Wrap(raw, bits, IntegerRange(format).Min < 0);
    }

    private static double RoundBFloat16(float x)
    {
        if (float.IsNaN(x))
        {
            return x;
        }

        uint bits = BitConverter.SingleToUInt32Bits(x);
        bits += 0x7FFF + ((bits >> 16) & 1);
        return BitConverter.UInt32BitsToSingle(bits & 0xFFFF0000);
    }

    private static double RoundMini(MiniFloat f, double x, bool saturate)
    {
        if (double.IsNaN(x))
        {
            return f.Specials == Specials.Finite ? -0.0 : double.NaN;
        }

        double a = Math.Abs(x);
        double r;
        if (double.IsInfinity(a))
        {
            r = double.PositiveInfinity;
        }
        else
        {
            int emin = 1 - f.Bias;
            int e = a == 0 ? emin : Math.Max(Math.ILogB(a), emin);
            double quantum = Math.ScaleB(1.0, e - f.MantissaBits);
            r = Math.Round(a / quantum, MidpointRounding.ToEven) * quantum;
        }

        if (r > f.Max)
        {
            r = saturate || f.Specials == Specials.Finite ? f.Max : f.Specials == Specials.Ieee ? double.PositiveInfinity : double.NaN;
        }

        if (r == 0 && f.Specials == Specials.Fnuz)
        {
            return 0.0;
        }

        return x < 0 || (x == 0 && double.IsNegative(x)) ? -r : r;
    }

    private static int EncodeMini(MiniFloat f, double r)
    {
        int sign = double.IsNegative(r) ? f.SignBit : 0;
        if (double.IsNaN(r))
        {
            return f.Specials switch
            {
                Specials.Fnuz => f.SignBit,
                Specials.FiniteOrNaN => f.SignBit - 1,
                Specials.Ieee => (f.MaxExponentField << f.MantissaBits) | (1 << (f.MantissaBits - 1)),
                _ => f.SignBit,
            };
        }

        double a = Math.Abs(r);
        if (double.IsInfinity(a))
        {
            return sign | (f.MaxExponentField << f.MantissaBits);
        }

        if (a == 0)
        {
            return sign;
        }

        int emin = 1 - f.Bias;
        int e = Math.ILogB(a);
        if (e < emin)
        {
            return sign | (int)(a / Math.ScaleB(1.0, emin - f.MantissaBits));
        }

        int mantissa = (int)(a / Math.ScaleB(1.0, e - f.MantissaBits)) - (1 << f.MantissaBits);
        return sign | ((e + f.Bias) << f.MantissaBits) | mantissa;
    }

    private static double DecodeMini(MiniFloat f, int code)
    {
        code &= (f.SignBit << 1) - 1;
        if (f.Specials == Specials.Fnuz && code == f.SignBit)
        {
            return double.NaN;
        }

        int exponent = (code >> f.MantissaBits) & f.MaxExponentField;
        int mantissa = code & ((1 << f.MantissaBits) - 1);
        bool negative = (code & f.SignBit) != 0;
        if (f.Specials == Specials.Ieee && exponent == f.MaxExponentField)
        {
            return mantissa != 0 ? double.NaN : negative ? double.NegativeInfinity : double.PositiveInfinity;
        }

        if (f.Specials == Specials.FiniteOrNaN && exponent == f.MaxExponentField && mantissa == (1 << f.MantissaBits) - 1)
        {
            return double.NaN;
        }

        double magnitude = exponent == 0
            ? mantissa * Math.ScaleB(1.0, 1 - f.Bias - f.MantissaBits)
            : (1 + mantissa * Math.ScaleB(1.0, -f.MantissaBits)) * Math.ScaleB(1.0, exponent - f.Bias);
        return negative ? -magnitude : magnitude;
    }

    /// <summary>E8M0 code of an f32 value (the exponent field, rounded as onnx.numpy_helper.to_float8e8m0 does).</summary>
    private static int EncodeE8M0(float x, bool saturate, ExponentRounding rounding)
    {
        uint bits = BitConverter.SingleToUInt32Bits(x);
        int exponent = (int)((bits >> 23) & 0xFF);
        if (exponent == 0xFF)
        {
            return 0xFF;
        }

        bool roundUp = rounding switch
        {
            ExponentRounding.Up => (bits & 0x7FFFFF) != 0,
            ExponentRounding.Nearest => (bits & 0x400000) != 0 && ((bits & 0x200000) != 0 || (bits & 0x1FFFFF) != 0 || exponent > 0),
            _ => false,
        };
        if (roundUp && !(saturate && exponent == 0xFE))
        {
            exponent++;
        }

        return exponent;
    }
}
