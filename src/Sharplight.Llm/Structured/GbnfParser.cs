using System.Globalization;

namespace Sharplight.Llm.Structured;

/// <summary>
/// Parser for GBNF, the grammar format of llama.cpp: <c>name ::= alternatives</c> rules with string literals, character
/// classes, <c>.</c>, groups, rule references and the quantifiers <c>* + ? {m} {m,} {m,n}</c>. Comments start with
/// <c>#</c>. A rule body runs until the next <c>name ::=</c>, so line breaks are free.
/// </summary>
internal sealed class GbnfParser
{
    private readonly string _text;
    private readonly string _prefix;
    private int _pos;

    private GbnfParser(string text, string prefix)
    {
        _text = text;
        _prefix = prefix;
    }

    /// <exception cref="FormatException">The text is not valid GBNF.</exception>
    public static GrammarBuilder Parse(string text)
    {
        var builder = new GrammarBuilder();
        ParseInto(builder, text, "");
        return builder;
    }

    /// <summary>Adds the rules of <paramref name="text"/> to <paramref name="builder"/>, every name prefixed with <paramref name="prefix"/>.</summary>
    /// <exception cref="FormatException">The text is not valid GBNF.</exception>
    public static void ParseInto(GrammarBuilder builder, string text, string prefix)
    {
        var parser = new GbnfParser(text, prefix);
        parser.SkipSpace();
        while (!parser.AtEnd)
        {
            string name = parser.ReadName();
            parser.SkipSpace();
            parser.Expect("::=");
            builder.Add(prefix + name, parser.ParseAlternatives(nested: false));
        }
    }

    private bool AtEnd => _pos >= _text.Length;

    private char Peek => _text[_pos];

    private FormatException Error(string message)
    {
        int line = 1 + _text.AsSpan(0, Math.Min(_pos, _text.Length)).Count('\n');
        return new FormatException($"GBNF line {line}: {message}.");
    }

    private void SkipSpace()
    {
        while (!AtEnd)
        {
            if (char.IsWhiteSpace(Peek))
            {
                _pos++;
            }
            else if (Peek == '#')
            {
                while (!AtEnd && Peek != '\n')
                {
                    _pos++;
                }
            }
            else
            {
                return;
            }
        }
    }

    private void Expect(string token)
    {
        if (!_text.AsSpan(_pos).StartsWith(token, StringComparison.Ordinal))
        {
            throw Error($"expected '{token}'");
        }

        _pos += token.Length;
        SkipSpace();
    }

    private static bool IsNameChar(char c) => char.IsAsciiLetterOrDigit(c) || c is '-' or '_';

    private string ReadName()
    {
        int start = _pos;
        while (!AtEnd && IsNameChar(Peek))
        {
            _pos++;
        }

        if (_pos == start)
        {
            throw Error("expected a rule name");
        }

        return _text[start.._pos];
    }

    /// <summary>True when the input continues with <c>name ::=</c> (the start of the next rule).</summary>
    private bool AtRuleStart()
    {
        int p = _pos;
        while (p < _text.Length && IsNameChar(_text[p]))
        {
            p++;
        }

        if (p == _pos)
        {
            return false;
        }

        while (p < _text.Length && char.IsWhiteSpace(_text[p]))
        {
            p++;
        }

        return _text.AsSpan(p).StartsWith("::=", StringComparison.Ordinal);
    }

    private GrammarExpr ParseAlternatives(bool nested)
    {
        var alternatives = new List<GrammarExpr> { ParseSequence(nested) };
        while (!AtEnd && Peek == '|')
        {
            _pos++;
            SkipSpace();
            alternatives.Add(ParseSequence(nested));
        }

        return GrammarExpr.Choice(alternatives);
    }

    private GrammarExpr ParseSequence(bool nested)
    {
        var items = new List<GrammarExpr>();
        while (!AtEnd && Peek is not ('|' or ')') && (nested || !AtRuleStart()))
        {
            var atom = ParseAtom();
            SkipSpace();
            items.Add(ParseQuantifiers(atom));
        }

        return GrammarExpr.Seq(items);
    }

    private GrammarExpr ParseQuantifiers(GrammarExpr atom)
    {
        while (!AtEnd)
        {
            switch (Peek)
            {
                case '*':
                    atom = GrammarExpr.Star(atom);
                    break;
                case '+':
                    atom = GrammarExpr.Plus(atom);
                    break;
                case '?':
                    atom = GrammarExpr.Optional(atom);
                    break;
                case '{':
                    atom = ParseBraces(atom);
                    continue;
                default:
                    return atom;
            }

            _pos++;
            SkipSpace();
        }

        return atom;
    }

    private GrammarExpr ParseBraces(GrammarExpr atom)
    {
        int close = _text.IndexOf('}', _pos);
        if (close < 0)
        {
            throw Error("unterminated '{'");
        }

        var body = _text.AsSpan(_pos + 1, close - _pos - 1).Trim();
        int comma = body.IndexOf(',');
        int min;
        int max;
        bool ok = int.TryParse(comma < 0 ? body : body[..comma].Trim(), NumberStyles.None, CultureInfo.InvariantCulture, out min);
        if (comma < 0)
        {
            max = min;
        }
        else if (body[(comma + 1)..].Trim().IsEmpty)
        {
            max = -1;
        }
        else
        {
            ok &= int.TryParse(body[(comma + 1)..].Trim(), NumberStyles.None, CultureInfo.InvariantCulture, out max) && max >= min;
        }

        if (!ok)
        {
            throw Error("invalid repetition bounds");
        }

        _pos = close + 1;
        SkipSpace();
        return GrammarExpr.Repeat(atom, min, max);
    }

    private GrammarExpr ParseAtom()
    {
        char c = Peek;
        switch (c)
        {
            case '"':
                return ParseLiteral();
            case '[':
                return ParseClass();
            case '.':
                _pos++;
                return CharClass.Any().ToExpr();
            case '(':
            {
                _pos++;
                SkipSpace();
                var inner = ParseAlternatives(nested: true);
                if (AtEnd || Peek != ')')
                {
                    throw Error("missing ')'");
                }

                _pos++;
                return inner;
            }

            default:
                if (IsNameChar(c))
                {
                    return GrammarExpr.Ref(_prefix + ReadName());
                }

                throw Error($"unexpected '{c}'");
        }
    }

    private GrammarExpr ParseLiteral()
    {
        _pos++;
        var sb = new System.Text.StringBuilder();
        while (true)
        {
            if (AtEnd || Peek == '\n')
            {
                throw Error("unterminated string literal");
            }

            if (Peek == '"')
            {
                _pos++;
                return GrammarExpr.Literal(sb.ToString());
            }

            sb.Append(char.ConvertFromUtf32(ReadChar()));
        }
    }

    private GrammarExpr ParseClass()
    {
        _pos++;
        bool negate = !AtEnd && Peek == '^';
        if (negate)
        {
            _pos++;
        }

        var cls = new CharClass();
        while (true)
        {
            if (AtEnd)
            {
                throw Error("unterminated character class");
            }

            if (Peek == ']')
            {
                _pos++;
                break;
            }

            int lo = ReadChar();
            if (_pos + 1 < _text.Length && Peek == '-' && _text[_pos + 1] != ']')
            {
                _pos++;
                cls.Add(lo, ReadChar());
            }
            else
            {
                cls.Add(lo);
            }
        }

        return (negate ? cls.Negate() : cls).ToExpr();
    }

    /// <summary>One code point of a literal or class, resolving escapes.</summary>
    private int ReadChar()
    {
        char c = _text[_pos++];
        if (c != '\\')
        {
            if (char.IsHighSurrogate(c) && !AtEnd && char.IsLowSurrogate(Peek))
            {
                return char.ConvertToUtf32(c, _text[_pos++]);
            }

            return c;
        }

        if (AtEnd)
        {
            throw Error("trailing backslash");
        }

        char e = _text[_pos++];
        return e switch
        {
            'n' => '\n',
            't' => '\t',
            'r' => '\r',
            'x' => ReadHex(2),
            'u' => ReadHex(4),
            'U' => ReadHex(8),
            _ => e,
        };
    }

    private int ReadHex(int digits)
    {
        if (_pos + digits > _text.Length
            || !int.TryParse(_text.AsSpan(_pos, digits), NumberStyles.HexNumber, CultureInfo.InvariantCulture, out int value))
        {
            throw Error("invalid hex escape");
        }

        _pos += digits;
        return value;
    }
}
