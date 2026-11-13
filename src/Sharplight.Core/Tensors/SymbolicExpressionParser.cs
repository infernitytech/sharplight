using System.Globalization;

namespace Sharplight.Tensors;

/// <summary>Recursive-descent parser for the text form of <see cref="SymbolicExpression"/>.</summary>
internal sealed class SymbolicExpressionParser(string text)
{
    private int _pos;

    public SymbolicExpression ParseAll()
    {
        var e = ParseSum();
        SkipSpace();
        if (_pos != text.Length)
        {
            throw Error("unexpected character");
        }

        return e;
    }

    private SymbolicExpression ParseSum()
    {
        var e = ParseProduct();
        while (true)
        {
            SkipSpace();
            if (Accept('+'))
            {
                e += ParseProduct();
            }
            else if (Accept('-'))
            {
                e -= ParseProduct();
            }
            else
            {
                return e;
            }
        }
    }

    private SymbolicExpression ParseProduct()
    {
        var e = ParseUnary();
        while (true)
        {
            SkipSpace();
            if (!Accept('*'))
            {
                return e;
            }

            e *= ParseUnary();
        }
    }

    private SymbolicExpression ParseUnary()
    {
        SkipSpace();
        return Accept('-') ? -ParseUnary() : ParsePrimary();
    }

    private SymbolicExpression ParsePrimary()
    {
        SkipSpace();
        if (Accept('('))
        {
            var inner = ParseSum();
            Expect(')');
            return inner;
        }

        if (_pos < text.Length && char.IsAsciiDigit(text[_pos]))
        {
            return SymbolicExpression.Constant(ParseInteger());
        }

        int start = _pos;
        while (_pos < text.Length && SymbolicExpression.IsNameChar(text[_pos]))
        {
            _pos++;
        }

        if (start == _pos)
        {
            throw Error("expected a number, symbol or '('");
        }

        string name = text[start.._pos];
        SkipSpace();
        if (!Accept('('))
        {
            return SymbolicExpression.Symbol(name);
        }

        return name switch
        {
            "floor" or "ceil" => ParseDivision(name == "ceil"),
            "min" or "max" => ParseExtremum(name == "max"),
            _ => throw Error($"unknown function '{name}'"),
        };
    }

    private SymbolicExpression ParseDivision(bool ceiling)
    {
        var operand = ParseSum();
        SkipSpace();
        Expect('/');
        SkipSpace();
        long divisor = ParseInteger();
        Expect(')');
        return ceiling ? operand.CeilDiv(divisor) : operand.FloorDiv(divisor);
    }

    private SymbolicExpression ParseExtremum(bool maximum)
    {
        var a = ParseSum();
        Expect(',');
        var b = ParseSum();
        Expect(')');
        return maximum ? SymbolicExpression.Max(a, b) : SymbolicExpression.Min(a, b);
    }

    private long ParseInteger()
    {
        int start = _pos;
        while (_pos < text.Length && char.IsAsciiDigit(text[_pos]))
        {
            _pos++;
        }

        if (start == _pos)
        {
            throw Error("expected an integer");
        }

        return long.Parse(text.AsSpan(start, _pos - start), NumberStyles.None, CultureInfo.InvariantCulture);
    }

    private void SkipSpace()
    {
        while (_pos < text.Length && text[_pos] == ' ')
        {
            _pos++;
        }
    }

    private bool Accept(char c)
    {
        if (_pos < text.Length && text[_pos] == c)
        {
            _pos++;
            return true;
        }

        return false;
    }

    private void Expect(char c)
    {
        SkipSpace();
        if (!Accept(c))
        {
            throw Error($"expected '{c}'");
        }
    }

    private FormatException Error(string message) => new($"Invalid symbolic expression '{text}' at {_pos}: {message}.");
}
