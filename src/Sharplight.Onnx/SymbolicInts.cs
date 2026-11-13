using System.Collections.Concurrent;
using Sharplight.Tensors;

namespace Sharplight.Onnx;

/// <summary>
/// Symbolic elements of integer constants. Shape computations (<c>Shape</c>, <c>Gather</c>, <c>Mul</c>, <c>Concat</c>, ...) fold at import
/// time even when a dimension is symbolic: such an element holds a reserved code that stands for a <see cref="SymbolicExpression"/>
/// (<c>N</c>, <c>12*N</c>, <c>N+1</c>). Data movement copies codes like ordinary values; arithmetic decodes, computes on the expressions
/// and encodes the result (a constant result becomes an ordinary value again).
/// </summary>
internal static class SymbolicInts
{
    // Far below any real shape value (and far from long.MinValue, which some ONNX exporters use as "minus infinity" in Slice).
    private const long Base = long.MinValue + (1L << 40);
    private const long Capacity = 1L << 24;

    // A symbolic element of a floating point constant (a shape cast to float) is a quiet NaN with this marker and the code index as payload.
    private const long DoubleMarker = 0x7FFC_0000_0000_0000;

    private static readonly ConcurrentDictionary<string, long> s_codes = new(StringComparer.Ordinal);
    private static readonly ConcurrentDictionary<long, SymbolicExpression> s_expressions = new();
    private static long s_next;

    /// <summary>Whether <paramref name="value"/> is the code of a symbolic expression.</summary>
    public static bool IsSymbolic(long value) => value >= Base && value < Base + Capacity;

    /// <summary>The value as a double; a symbolic code becomes a NaN that carries it (see <see cref="FromDouble"/>).</summary>
    public static double ToDouble(long value) => IsSymbolic(value) ? BitConverter.Int64BitsToDouble(DoubleMarker | (value - Base)) : value;

    /// <summary>Whether a floating point element carries a symbolic code.</summary>
    public static bool IsSymbolicDouble(double value) => double.IsNaN(value) && (BitConverter.DoubleToInt64Bits(value) & ~(Capacity - 1)) == DoubleMarker;

    /// <summary>The code carried by a floating point element, or the element truncated to an integer.</summary>
    public static long FromDouble(double value) => IsSymbolicDouble(value) ? Base + (BitConverter.DoubleToInt64Bits(value) & (Capacity - 1)) : (long)value;

    /// <summary>Whether any element of <paramref name="values"/> is symbolic.</summary>
    public static bool AnySymbolic(IEnumerable<long> values) => values.Any(IsSymbolic);

    /// <summary>The value of a constant expression, or the code of a symbolic one.</summary>
    public static long Encode(SymbolicExpression e)
    {
        if (e.IsConstant)
        {
            return e.ConstantValue;
        }

        return s_codes.GetOrAdd(e.ToString(), _ =>
        {
            long code = Base + Interlocked.Increment(ref s_next);
            if (code >= Base + Capacity)
            {
                throw new InvalidOperationException("Too many distinct symbolic shape values.");
            }

            s_expressions[code] = e;
            return code;
        });
    }

    /// <summary>The encoded extent of a dimension.</summary>
    public static long Encode(Dim d) => d.IsSymbolic ? Encode(d.Expression) : d.Fixed;

    /// <summary>The expression of a code, or the constant expression of an ordinary value.</summary>
    public static SymbolicExpression Decode(long value) => IsSymbolic(value) ? s_expressions[value] : SymbolicExpression.Constant(value);

    /// <summary>The dimension an encoded extent stands for.</summary>
    public static Dim ToDim(long value) => Dim.FromExpression(Decode(value));

    /// <summary>Readable form of an encoded value (for error messages).</summary>
    public static string Format(long value) => Decode(value).ToString();

    /// <summary>
    /// Decides a comparison of two encoded values whose difference may be symbolic, assuming every symbol is at least 1.
    /// </summary>
    /// <returns>The sign of <c>a - b</c>.</returns>
    /// <exception cref="NotSupportedException">The sign depends on the symbol values.</exception>
    public static int CompareSigned(long a, long b)
    {
        var difference = Decode(a) - Decode(b);
        if (difference.IsConstant)
        {
            return Math.Sign(difference.ConstantValue);
        }

        long atOne = difference.Evaluate(_ => 1);
        long atLarge = difference.Evaluate(_ => 1L << 20);
        if (atOne > 0 && atLarge > 0 && IsMonotone(difference, increasing: true))
        {
            return 1;
        }

        if (atOne < 0 && atLarge < 0 && IsMonotone(difference, increasing: false))
        {
            return -1;
        }

        throw new NotSupportedException($"Comparing {Decode(a)} with {Decode(b)} depends on the symbol values.");
    }

    // Every factor kind is non-decreasing in its symbols, so the sign of all coefficients decides monotonicity.
    private static bool IsMonotone(SymbolicExpression e, bool increasing)
    {
        long baseValue = e.Evaluate(_ => 1);
        foreach (var atom in e.Atoms())
        {
            long bumped = e.Evaluate(s => s == atom ? 1L << 20 : 1);
            if (increasing ? bumped < baseValue : bumped > baseValue)
            {
                return false;
            }
        }

        return true;
    }
}
