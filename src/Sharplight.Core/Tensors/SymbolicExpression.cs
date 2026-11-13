using System.Collections.Concurrent;
using System.Globalization;
using System.Text;

namespace Sharplight.Tensors;

/// <summary>
/// An integer expression over named symbols, used for dimensions such as <c>12*N</c>, <c>N+1</c> or <c>ceil(N/32)</c>.
/// Expressions are kept in a normal form (a sum of monomials with integer coefficients; division, minimum and maximum
/// are opaque factors), so the printed text is canonical and serves as the identity: equal expressions print equally.
/// </summary>
/// <remarks>
/// Text syntax: integers, symbol names (<c>[A-Za-z_][A-Za-z0-9_.#]*</c>), <c>+ - *</c>, parentheses,
/// <c>floor(e/c)</c>, <c>ceil(e/c)</c>, <c>min(a,b)</c> and <c>max(a,b)</c> with a positive integer divisor <c>c</c>.
/// Instances are immutable and thread-safe.
/// </remarks>
public sealed class SymbolicExpression : IEquatable<SymbolicExpression>
{
    private static readonly ConcurrentDictionary<string, SymbolicExpression> s_parsed = new(StringComparer.Ordinal);

    private readonly (long Coefficient, SymbolicFactor[] Factors, string Key)[] _terms;
    private readonly long _constant;
    private readonly string _text;

    private SymbolicExpression((long Coefficient, SymbolicFactor[] Factors, string Key)[] terms, long constant)
    {
        _terms = terms;
        _constant = constant;
        _text = Format(terms, constant);
    }

    /// <summary>True when the expression does not depend on any symbol.</summary>
    public bool IsConstant => _terms.Length == 0;

    /// <summary>The value of a constant expression.</summary>
    /// <exception cref="InvalidOperationException">The expression depends on a symbol.</exception>
    public long ConstantValue => IsConstant ? _constant : throw new InvalidOperationException($"'{this}' is not constant.");

    /// <summary>True when the expression is a single symbol (coefficient 1, no constant).</summary>
    public bool IsSymbol => _constant == 0 && _terms.Length == 1 && _terms[0].Coefficient == 1
        && _terms[0].Factors.Length == 1 && _terms[0].Factors[0].Kind == SymbolicFactorKind.Symbol;

    // ---- construction ------------------------------------------------------------------------
    /// <summary>A constant expression.</summary>
    public static SymbolicExpression Constant(long value) => new([], value);

    /// <summary>
    /// A single symbol.
    /// </summary>
    /// <param name="name">Symbol name (letters, digits, <c>_ . #</c>; not starting with a digit).</param>
    /// <exception cref="ArgumentException"><paramref name="name"/> is not a valid symbol name.</exception>
    public static SymbolicExpression Symbol(string name)
    {
        if (!IsSymbolName(name))
        {
            throw new ArgumentException($"'{name}' is not a valid symbol name.", nameof(name));
        }

        return Monomial(1, [SymbolicFactor.Atom(name)]);
    }

    /// <summary>
    /// Parses an expression (results are cached by text).
    /// </summary>
    /// <param name="text">Expression text, for example <c>12*N+1</c>.</param>
    /// <exception cref="FormatException">The text is not a valid expression.</exception>
    public static SymbolicExpression Parse(string text) => s_parsed.GetOrAdd(text, static t => new SymbolicExpressionParser(t).ParseAll());

    /// <summary>Whether <paramref name="text"/> is a plain symbol name rather than a compound expression.</summary>
    public static bool IsSymbolName(string text)
    {
        if (text.Length == 0 || !(char.IsAsciiLetter(text[0]) || text[0] == '_'))
        {
            return false;
        }

        foreach (char c in text)
        {
            if (!IsNameChar(c))
            {
                return false;
            }
        }

        return true;
    }

    internal static bool IsNameChar(char c) => char.IsAsciiLetterOrDigit(c) || c is '_' or '.' or '#';

    private static SymbolicExpression Monomial(long coefficient, SymbolicFactor[] factors)
    {
        Array.Sort(factors, (a, b) => string.CompareOrdinal(a.Text, b.Text));
        return coefficient == 0 ? Constant(0) : new([(coefficient, factors, KeyOf(factors))], 0);
    }

    private static string KeyOf(SymbolicFactor[] factors) => string.Join("*", factors.Select(f => f.Text));

    private static SymbolicExpression FromTerms(Dictionary<string, (long Coefficient, SymbolicFactor[] Factors)> terms, long constant)
    {
        var list = terms.Where(t => t.Value.Coefficient != 0)
            .OrderBy(t => t.Key, StringComparer.Ordinal)
            .Select(t => (t.Value.Coefficient, t.Value.Factors, t.Key))
            .ToArray();
        return new SymbolicExpression(list, constant);
    }

    private Dictionary<string, (long Coefficient, SymbolicFactor[] Factors)> TermMap()
        => _terms.ToDictionary(t => t.Key, t => (t.Coefficient, t.Factors), StringComparer.Ordinal);

    // ---- arithmetic --------------------------------------------------------------------------
    /// <summary>Sum.</summary>
    public static SymbolicExpression operator +(SymbolicExpression a, SymbolicExpression b)
    {
        var map = a.TermMap();
        foreach (var (c, f, k) in b._terms)
        {
            map[k] = map.TryGetValue(k, out var e) ? (e.Coefficient + c, e.Factors) : (c, f);
        }

        return FromTerms(map, a._constant + b._constant);
    }

    /// <summary>Negation.</summary>
    public static SymbolicExpression operator -(SymbolicExpression a) => a * -1;

    /// <summary>Difference.</summary>
    public static SymbolicExpression operator -(SymbolicExpression a, SymbolicExpression b) => a + -b;

    /// <summary>Product with a constant.</summary>
    public static SymbolicExpression operator *(SymbolicExpression a, long b)
        => b == 0 ? Constant(0) : new(a._terms.Select(t => (t.Coefficient * b, t.Factors, t.Key)).ToArray(), a._constant * b);

    /// <summary>Product (distributes over sums).</summary>
    public static SymbolicExpression operator *(SymbolicExpression a, SymbolicExpression b)
    {
        if (b.IsConstant)
        {
            return a * b._constant;
        }

        if (a.IsConstant)
        {
            return b * a._constant;
        }

        var result = Constant(a._constant * b._constant);
        foreach (var (c, f, _) in a._terms)
        {
            result += Monomial(c * b._constant, [.. f]);
        }

        foreach (var (c, f, _) in b._terms)
        {
            result += Monomial(c * a._constant, [.. f]);
        }

        foreach (var (ca, fa, _) in a._terms)
        {
            foreach (var (cb, fb, _) in b._terms)
            {
                result += Monomial(ca * cb, [.. fa, .. fb]);
            }
        }

        return result;
    }

    /// <summary>Implicit conversion of a constant.</summary>
    public static implicit operator SymbolicExpression(long value) => Constant(value);

    /// <summary>
    /// <c>floor(this / divisor)</c>. Terms divisible by the divisor are divided exactly; only the remainder stays inside an
    /// opaque <c>floor</c>, so <c>(12*N)/12</c> simplifies to <c>N</c>.
    /// </summary>
    /// <exception cref="DivideByZeroException"><paramref name="divisor"/> is 0.</exception>
    public SymbolicExpression FloorDiv(long divisor) => Divide(divisor, ceiling: false);

    /// <summary><c>ceil(this / divisor)</c>, simplified like <see cref="FloorDiv"/>.</summary>
    /// <exception cref="DivideByZeroException"><paramref name="divisor"/> is 0.</exception>
    public SymbolicExpression CeilDiv(long divisor) => Divide(divisor, ceiling: true);

    private SymbolicExpression Divide(long divisor, bool ceiling)
    {
        if (divisor == 0)
        {
            throw new DivideByZeroException($"Division of '{this}' by zero.");
        }

        if (divisor < 0)
        {
            return (-this).Divide(-divisor, ceiling);
        }

        long quotient = Math.DivRem(_constant, divisor, out long remainder);
        if (remainder < 0)
        {
            quotient--;
            remainder += divisor;
        }

        var exact = new Dictionary<string, (long Coefficient, SymbolicFactor[] Factors)>(StringComparer.Ordinal);
        var rest = new Dictionary<string, (long Coefficient, SymbolicFactor[] Factors)>(StringComparer.Ordinal);
        foreach (var (c, f, k) in _terms)
        {
            if (c % divisor == 0)
            {
                exact[k] = (c / divisor, f);
            }
            else
            {
                rest[k] = (c, f);
            }
        }

        var whole = FromTerms(exact, quotient);
        if (rest.Count == 0)
        {
            return remainder > 0 && ceiling ? whole + 1 : whole;
        }

        if (divisor == 1)
        {
            return whole + FromTerms(rest, remainder);
        }

        return whole + Monomial(1, [SymbolicFactor.Division(FromTerms(rest, remainder), divisor, ceiling)]);
    }

    /// <summary>Minimum; decided at build time when the operands differ by a constant.</summary>
    public static SymbolicExpression Min(SymbolicExpression a, SymbolicExpression b) => Extremum(a, b, maximum: false);

    /// <summary>Maximum; decided at build time when the operands differ by a constant.</summary>
    public static SymbolicExpression Max(SymbolicExpression a, SymbolicExpression b) => Extremum(a, b, maximum: true);

    private static SymbolicExpression Extremum(SymbolicExpression a, SymbolicExpression b, bool maximum)
    {
        var difference = a - b;
        if (difference.IsConstant)
        {
            return (difference._constant >= 0) == maximum ? a : b;
        }

        return Monomial(1, [SymbolicFactor.Extremum(a, b, maximum)]);
    }

    // ---- evaluation --------------------------------------------------------------------------
    /// <summary>
    /// Evaluates the expression.
    /// </summary>
    /// <param name="atom">Value of an atomic symbol; may throw for unbound symbols.</param>
    public long Evaluate(Func<string, long> atom)
    {
        long value = _constant;
        foreach (var (c, f, _) in _terms)
        {
            long product = c;
            foreach (var factor in f)
            {
                product *= factor.Evaluate(atom);
            }

            value += product;
        }

        return value;
    }

    /// <summary>
    /// Evaluates the expression with the given symbol bindings.
    /// </summary>
    /// <exception cref="InvalidOperationException">A symbol is not bound.</exception>
    public long Evaluate(IReadOnlyDictionary<string, long> bindings)
        => Evaluate(s => bindings.TryGetValue(s, out var v) ? v : throw new InvalidOperationException($"Symbolic dimension '{s}' is not bound."));

    /// <summary>
    /// Upper bound of the expression given upper bounds of its symbols (symbols are assumed non-negative; terms with a
    /// negative coefficient are bounded with their symbols at 0). Exact for expressions with non-negative coefficients.
    /// </summary>
    /// <param name="atomBound">Upper bound of an atomic symbol.</param>
    public long UpperBound(Func<string, long> atomBound)
    {
        long value = _constant;
        foreach (var (c, f, _) in _terms)
        {
            if (c < 0)
            {
                continue;
            }

            long product = c;
            foreach (var factor in f)
            {
                product *= factor.UpperBound(atomBound);
            }

            value += product;
        }

        return Math.Max(0, value);
    }

    /// <summary>The distinct atomic symbols the expression depends on.</summary>
    public IEnumerable<string> Atoms() => _terms.SelectMany(t => t.Factors.SelectMany(f => f.Atoms())).Distinct();

    /// <summary>
    /// The value of a symbol or expression name under <paramref name="bindings"/>: a direct binding when present, otherwise
    /// the expression evaluated from the bindings of its atoms. Providers use this wherever they look up a dimension by name.
    /// </summary>
    /// <exception cref="InvalidOperationException">A symbol is not bound.</exception>
    public static long Evaluate(string name, IReadOnlyDictionary<string, long> bindings)
    {
        if (bindings.TryGetValue(name, out var v))
        {
            return v;
        }

        if (IsSymbolName(name))
        {
            throw new InvalidOperationException($"Symbolic dimension '{name}' is not bound.");
        }

        return Parse(name).Evaluate(bindings);
    }

    /// <summary>Like <see cref="Evaluate(string, IReadOnlyDictionary{string, long})"/>, returning false when a symbol is unbound.</summary>
    public static bool TryEvaluate(string name, IReadOnlyDictionary<string, long> bindings, out long value)
    {
        if (bindings.TryGetValue(name, out value))
        {
            return true;
        }

        if (IsSymbolName(name))
        {
            return false;
        }

        var expression = Parse(name);
        if (expression.Atoms().Any(a => !bindings.ContainsKey(a)))
        {
            return false;
        }

        value = expression.Evaluate(bindings);
        return true;
    }

    // ---- text --------------------------------------------------------------------------------
    private static string Format((long Coefficient, SymbolicFactor[] Factors, string Key)[] terms, long constant)
    {
        if (terms.Length == 0)
        {
            return constant.ToString(CultureInfo.InvariantCulture);
        }

        var sb = new StringBuilder();
        foreach (var (c, _, key) in terms)
        {
            if (sb.Length > 0)
            {
                sb.Append(c < 0 ? '-' : '+');
            }
            else if (c < 0)
            {
                sb.Append('-');
            }

            long magnitude = Math.Abs(c);
            if (magnitude != 1)
            {
                sb.Append(magnitude.ToString(CultureInfo.InvariantCulture)).Append('*');
            }

            sb.Append(key);
        }

        if (constant != 0)
        {
            sb.Append(constant < 0 ? '-' : '+').Append(Math.Abs(constant).ToString(CultureInfo.InvariantCulture));
        }

        return sb.ToString();
    }

    /// <summary>The text with parentheses unless it is a single symbol or constant (for use as an operand).</summary>
    internal string ToOperandString() => IsSymbol || (IsConstant && _constant >= 0) ? _text : $"({_text})";

    /// <summary>The canonical text.</summary>
    public override string ToString() => _text;

    /// <inheritdoc />
    public bool Equals(SymbolicExpression? other) => other is not null && _text == other._text;

    /// <inheritdoc />
    public override bool Equals(object? obj) => Equals(obj as SymbolicExpression);

    /// <inheritdoc />
    public override int GetHashCode() => _text.GetHashCode(StringComparison.Ordinal);
}
