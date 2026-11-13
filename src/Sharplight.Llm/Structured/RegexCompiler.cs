using System.Globalization;

namespace Sharplight.Llm.Structured;

/// <summary>
/// Translates a regular expression into a <see cref="GrammarExpr"/> that matches its whole language: literals, escapes,
/// classes, <c>.</c>, groups, alternation and the usual quantifiers. Anchors and word boundaries are ignored (the whole
/// output is matched); back-references and look-arounds are rejected.
/// </summary>
internal sealed class RegexCompiler
{
    private readonly string _pattern;
    private int _pos;

    private RegexCompiler(string pattern)
    {
        _pattern = pattern;
    }

    /// <exception cref="FormatException">Invalid or unsupported pattern.</exception>
    public static GrammarExpr Compile(string pattern)
    {
        var compiler = new RegexCompiler(pattern);
        var expr = compiler.ParseAlternation();
        if (compiler._pos < pattern.Length)
        {
            throw compiler.Error("unbalanced ')'");
        }

        return expr;
    }

    private bool AtEnd => _pos >= _pattern.Length;

    private char Peek => _pattern[_pos];

    private FormatException Error(string message) => new($"Regex '{_pattern}' at {_pos}: {message}.");

    private GrammarExpr ParseAlternation()
    {
        var alternatives = new List<GrammarExpr> { ParseSequence() };
        while (!AtEnd && Peek == '|')
        {
            _pos++;
            alternatives.Add(ParseSequence());
        }

        return GrammarExpr.Choice(alternatives);
    }

    private GrammarExpr ParseSequence()
    {
        var items = new List<GrammarExpr>();
        while (!AtEnd && Peek is not ('|' or ')'))
        {
            var atom = ParseAtom();
            items.Add(ParseQuantifier(atom));
        }

        return GrammarExpr.Seq(items);
    }

    private GrammarExpr ParseQuantifier(GrammarExpr atom)
    {
        while (!AtEnd)
        {
            int min;
            int max;
            switch (Peek)
            {
                case '*':
                    (min, max) = (0, -1);
                    _pos++;
                    break;
                case '+':
                    (min, max) = (1, -1);
                    _pos++;
                    break;
                case '?':
                    (min, max) = (0, 1);
                    _pos++;
                    break;
                case '{' when TryParseBraces(out min, out max):
                    break;
                default:
                    return atom;
            }

            // Lazy and possessive markers do not change the language.
            if (!AtEnd && Peek is '?' or '+')
            {
                _pos++;
            }

            atom = GrammarExpr.Repeat(atom, min, max);
        }

        return atom;
    }

    /// <summary>Parses <c>{n}</c>, <c>{n,}</c>, <c>{n,m}</c>; a brace that does not start a quantifier is a literal.</summary>
    private bool TryParseBraces(out int min, out int max)
    {
        min = max = 0;
        int close = _pattern.IndexOf('}', _pos);
        if (close < 0)
        {
            return false;
        }

        var body = _pattern.AsSpan(_pos + 1, close - _pos - 1);
        int comma = body.IndexOf(',');
        var first = comma < 0 ? body : body[..comma];
        if (!int.TryParse(first, NumberStyles.None, CultureInfo.InvariantCulture, out min))
        {
            return false;
        }

        if (comma < 0)
        {
            max = min;
        }
        else if (body.Length == comma + 1)
        {
            max = -1;
        }
        else if (!int.TryParse(body[(comma + 1)..], NumberStyles.None, CultureInfo.InvariantCulture, out max) || max < min)
        {
            throw Error("invalid repetition bounds");
        }

        _pos = close + 1;
        return true;
    }

    private GrammarExpr ParseAtom()
    {
        char c = Peek;
        switch (c)
        {
            case '(':
                return ParseGroup();
            case '[':
                _pos++;
                return ParseClass().ToExpr();
            case '.':
                _pos++;
                return CharClass.AnyButNewline().ToExpr();
            case '^':
            case '$':
                _pos++;
                return GrammarExpr.Empty;
            case '\\':
                _pos++;
                return ParseEscapeAtom();
            case '*' or '+' or '?':
                throw Error($"nothing to repeat before '{c}'");
            default:
                return GrammarExpr.Literal(char.ConvertFromUtf32(ReadCodePoint()));
        }
    }

    private GrammarExpr ParseGroup()
    {
        _pos++;
        if (!AtEnd && Peek == '?')
        {
            _pos++;
            if (AtEnd)
            {
                throw Error("unterminated group");
            }

            switch (Peek)
            {
                case ':':
                    _pos++;
                    break;
                case 'P' or '<' when !_pattern.AsSpan(_pos).StartsWith("<=") && !_pattern.AsSpan(_pos).StartsWith("<!"):
                {
                    int close = _pattern.IndexOf('>', _pos);
                    if (close < 0)
                    {
                        throw Error("unterminated group name");
                    }

                    _pos = close + 1;
                    break;
                }

                default:
                    throw Error("look-arounds, flags and other group extensions are not supported");
            }
        }

        var inner = ParseAlternation();
        if (AtEnd || Peek != ')')
        {
            throw Error("missing ')'");
        }

        _pos++;
        return inner;
    }

    private GrammarExpr ParseEscapeAtom()
    {
        if (AtEnd)
        {
            throw Error("trailing backslash");
        }

        char c = Peek;
        if (c is 'b' or 'B' or 'A' or 'z' or 'Z')
        {
            _pos++;
            return GrammarExpr.Empty;   // assertions: the whole output is matched anyway
        }

        if (c is >= '1' and <= '9')
        {
            throw Error("back-references are not supported");
        }

        return ParseEscape().ToExpr();
    }

    /// <summary>Parses the escape after a backslash into the class it denotes.</summary>
    private CharClass ParseEscape()
    {
        char c = _pattern[_pos++];
        switch (c)
        {
            case 'd':
                return new CharClass().Add('0', '9');
            case 'D':
                return new CharClass().Add('0', '9').Negate();
            case 'w':
                return Word();
            case 'W':
                return Word().Negate();
            case 's':
                return Space();
            case 'S':
                return Space().Negate();
            default:
                return new CharClass().Add(EscapedCodePoint(c));
        }
    }

    private static CharClass Word() => new CharClass().Add('a', 'z').Add('A', 'Z').Add('0', '9').Add('_');

    private static CharClass Space() => new CharClass().Add(' ').Add('\t', '\r');

    private int EscapedCodePoint(char c) => c switch
    {
        'n' => '\n',
        't' => '\t',
        'r' => '\r',
        'f' => '\f',
        'v' => '\v',
        '0' => 0,
        'x' => ReadHex(2),
        'u' => ReadHex(4),
        _ when char.IsAsciiLetterOrDigit(c) => throw Error($"unsupported escape '\\{c}'"),
        _ => c,
    };

    private int ReadHex(int digits)
    {
        if (_pos + digits > _pattern.Length
            || !int.TryParse(_pattern.AsSpan(_pos, digits), NumberStyles.HexNumber, CultureInfo.InvariantCulture, out int value))
        {
            throw Error("invalid hex escape");
        }

        _pos += digits;
        return value;
    }

    private int ReadCodePoint()
    {
        if (char.IsHighSurrogate(_pattern[_pos]) && _pos + 1 < _pattern.Length)
        {
            int cp = char.ConvertToUtf32(_pattern[_pos], _pattern[_pos + 1]);
            _pos += 2;
            return cp;
        }

        return _pattern[_pos++];
    }

    private CharClass ParseClass()
    {
        bool negate = !AtEnd && Peek == '^';
        if (negate)
        {
            _pos++;
        }

        var cls = new CharClass();
        bool first = true;
        while (true)
        {
            if (AtEnd)
            {
                throw Error("unterminated character class");
            }

            if (Peek == ']' && !first)
            {
                _pos++;
                break;
            }

            first = false;
            int lo;
            if (Peek == '\\')
            {
                _pos++;
                if (AtEnd)
                {
                    throw Error("trailing backslash");
                }

                if (Peek is 'd' or 'D' or 'w' or 'W' or 's' or 'S')
                {
                    cls.Add(ParseEscape());
                    continue;
                }

                lo = EscapedCodePoint(_pattern[_pos++]);
            }
            else
            {
                lo = ReadCodePoint();
            }

            if (_pos + 1 < _pattern.Length && Peek == '-' && _pattern[_pos + 1] != ']')
            {
                _pos++;
                int hi = Peek == '\\' ? EscapedAfterBackslash() : ReadCodePoint();
                cls.Add(lo, hi);
            }
            else
            {
                cls.Add(lo);
            }
        }

        return negate ? cls.Negate() : cls;
    }

    private int EscapedAfterBackslash()
    {
        _pos++;
        if (AtEnd)
        {
            throw Error("trailing backslash");
        }

        return EscapedCodePoint(_pattern[_pos++]);
    }
}
