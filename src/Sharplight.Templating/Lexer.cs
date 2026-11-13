using System.Globalization;
using System.Text;

namespace Sharplight.Templating;

internal enum TokenKind
{
    Data,
    VarBegin,
    VarEnd,
    BlockBegin,
    BlockEnd,
    Name,
    String,
    Int,
    Float,
    Op,
    Eof,
}

internal sealed class Token
{
    public TokenKind Kind;
    public string Text = "";
    public object? Value;
    public int Line, Col;

    public override string ToString() => Kind switch
    {
        TokenKind.Eof => "end of template",
        TokenKind.Data => "text",
        TokenKind.VarBegin => "'{{'",
        TokenKind.VarEnd => "'}}'",
        TokenKind.BlockBegin => "'{%'",
        TokenKind.BlockEnd => "'%}'",
        TokenKind.String => "string",
        _ => $"'{Text}'",
    };
}

/// <summary>Turns template source into tokens, applying whitespace control, trim_blocks and lstrip_blocks.</summary>
internal sealed class Lexer
{
    private readonly string _s;
    private readonly bool _trimBlocks, _lstripBlocks;
    private readonly List<Token> _tokens = new();
    private readonly List<int> _lineStarts = new() { 0 };

    private Lexer(string source, bool trimBlocks, bool lstripBlocks)
    {
        _s = source;
        _trimBlocks = trimBlocks;
        _lstripBlocks = lstripBlocks;
        for (int i = 0; i < _s.Length; i++)
            if (_s[i] == '\n') _lineStarts.Add(i + 1);
    }

    public static List<Token> Tokenize(string source, bool trimBlocks, bool lstripBlocks, bool keepTrailingNewline)
    {
        source = source.Replace("\r\n", "\n").Replace('\r', '\n');
        if (!keepTrailingNewline && source.EndsWith('\n')) source = source.Substring(0, source.Length - 1);
        var lx = new Lexer(source, trimBlocks, lstripBlocks);
        lx.Run();
        return lx._tokens;
    }

    private (int Line, int Col) Pos(int index)
    {
        int lo = 0, hi = _lineStarts.Count - 1;
        while (lo < hi)
        {
            int mid = (lo + hi + 1) / 2;
            if (_lineStarts[mid] <= index) lo = mid;
            else hi = mid - 1;
        }
        return (lo + 1, index - _lineStarts[lo] + 1);
    }

    private TemplateSyntaxException Error(string msg, int index)
    {
        var (l, c) = Pos(index);
        return new TemplateSyntaxException(msg, l, c);
    }

    private void Add(TokenKind kind, string text, int index, object? value = null)
    {
        var (l, c) = Pos(index);
        _tokens.Add(new Token { Kind = kind, Text = text, Value = value, Line = l, Col = c });
    }

    private int FindTagStart(int from)
    {
        while (true)
        {
            int i = _s.IndexOf('{', from);
            if (i < 0 || i + 1 >= _s.Length) return -1;
            char n = _s[i + 1];
            if (n == '{' || n == '%' || n == '#') return i;
            from = i + 1;
        }
    }

    private void AddData(int from, int to, bool lstripStart, bool rstripEnd, bool lstripBlock)
    {
        if (to <= from) return;
        string text = _s.Substring(from, to - from);
        if (lstripStart) text = text.TrimStart();
        if (rstripEnd) text = text.TrimEnd();
        else if (lstripBlock)
        {
            int l = text.LastIndexOf('\n') + 1;
            if (l > 0 || from == 0 || _s[from - 1] == '\n')
            {
                bool blank = true;
                for (int i = l; i < text.Length; i++)
                    if (text[i] != ' ' && text[i] != '\t') { blank = false; break; }
                if (blank) text = text.Substring(0, l);
            }
        }
        if (text.Length > 0) Add(TokenKind.Data, text, from, text);
    }

    private void Run()
    {
        int pos = 0;
        bool lstripNext = false;
        while (true)
        {
            int t = FindTagStart(pos);
            if (t < 0)
            {
                AddData(pos, _s.Length, lstripNext, false, false);
                break;
            }
            char kind = _s[t + 1];
            int p = t + 2;
            bool rstripPrev = false, noLstrip = false;
            if (p < _s.Length && _s[p] == '-')
            {
                rstripPrev = true;
                p++;
            }
            else if (p < _s.Length && _s[p] == '+' && kind != '{')
            {
                noLstrip = true;
                p++;
            }
            AddData(pos, t, lstripNext, rstripPrev, _lstripBlocks && kind != '{' && !noLstrip);
            lstripNext = false;

            if (kind == '#')
            {
                int e = _s.IndexOf("#}", p, StringComparison.Ordinal);
                if (e < 0) throw Error("Missing end of comment tag", t);
                if (e > p && _s[e - 1] == '-') lstripNext = true;
                pos = e + 2;
                if (_trimBlocks && pos < _s.Length && _s[pos] == '\n') pos++;
                continue;
            }

            bool isBlock = kind == '%';
            Add(isBlock ? TokenKind.BlockBegin : TokenKind.VarBegin, isBlock ? "{%" : "{{", t);
            int firstInner = _tokens.Count;
            bool rstripNext = LexInner(ref p, isBlock, t);
            pos = p;
            lstripNext = rstripNext;
            bool isRaw = isBlock && _tokens.Count == firstInner + 2 && _tokens[firstInner].Kind == TokenKind.Name && _tokens[firstInner].Text == "raw" && _tokens[firstInner + 1].Kind == TokenKind.BlockEnd;
            if (isBlock && !isRaw && _trimBlocks && !rstripNext && pos < _s.Length && _s[pos] == '\n') pos++;

            // raw block: [BlockBegin, Name raw, BlockEnd]; trim_blocks does not apply to the raw tag itself
            if (isRaw)
            {
                _tokens.RemoveRange(firstInner - 1, 3);
                int search = pos;
                while (true)
                {
                    int b = _s.IndexOf("{%", search, StringComparison.Ordinal);
                    if (b < 0) throw Error("Missing end of raw directive", t);
                    if (MatchEndRaw(b, out int endIdx, out bool stripBefore, out bool stripAfter))
                    {
                        AddData(pos, b, lstripNext, stripBefore, false);
                        pos = endIdx;
                        lstripNext = stripAfter;
                        if (_trimBlocks && !stripAfter && pos < _s.Length && _s[pos] == '\n') pos++;
                        break;
                    }
                    search = b + 2;
                }
            }
        }
        Add(TokenKind.Eof, "", _s.Length);
    }

    private bool MatchEndRaw(int b, out int endIdx, out bool stripBefore, out bool stripAfter)
    {
        endIdx = 0;
        stripBefore = stripAfter = false;
        int i = b + 2;
        if (i < _s.Length && (_s[i] == '-' || _s[i] == '+'))
        {
            stripBefore = _s[i] == '-';
            i++;
        }
        while (i < _s.Length && char.IsWhiteSpace(_s[i])) i++;
        if (string.CompareOrdinal(_s, i, "endraw", 0, 6) != 0) return false;
        i += 6;
        while (i < _s.Length && char.IsWhiteSpace(_s[i])) i++;
        if (i < _s.Length && (_s[i] == '-' || _s[i] == '+'))
        {
            stripAfter = _s[i] == '-';
            i++;
        }
        if (i + 1 < _s.Length && _s[i] == '%' && _s[i + 1] == '}')
        {
            endIdx = i + 2;
            return true;
        }
        return false;
    }

    private static bool IsNameStart(char c) => c == '_' || char.IsLetter(c);
    private static bool IsNameChar(char c) => c == '_' || char.IsLetterOrDigit(c);

    /// <summary>Lexes expression tokens up to the closing delimiter. Returns true if the closer had a '-' marker.</summary>
    private bool LexInner(ref int p, bool isBlock, int tagStart)
    {
        string end = isBlock ? "%}" : "}}";
        int depth = 0;
        while (true)
        {
            while (p < _s.Length && char.IsWhiteSpace(_s[p])) p++;
            if (p >= _s.Length) throw Error(isBlock ? "Unexpected end of template, expected '%}'" : "Unexpected end of template, expected '}}'", tagStart);
            char c = _s[p];
            if (depth == 0)
            {
                if (c == '-' && string.CompareOrdinal(_s, p + 1, end, 0, 2) == 0)
                {
                    Add(isBlock ? TokenKind.BlockEnd : TokenKind.VarEnd, end, p);
                    p += 3;
                    return true;
                }
                if (c == end[0] && string.CompareOrdinal(_s, p, end, 0, 2) == 0)
                {
                    Add(isBlock ? TokenKind.BlockEnd : TokenKind.VarEnd, end, p);
                    p += 2;
                    return false;
                }
            }
            int start = p;
            if (IsNameStart(c))
            {
                while (p < _s.Length && IsNameChar(_s[p])) p++;
                Add(TokenKind.Name, _s.Substring(start, p - start), start);
            }
            else if (char.IsAsciiDigit(c))
            {
                while (p < _s.Length && (char.IsAsciiDigit(_s[p]) || _s[p] == '_')) p++;
                bool isFloat = false;
                if (p + 1 < _s.Length && _s[p] == '.' && char.IsAsciiDigit(_s[p + 1]))
                {
                    isFloat = true;
                    p++;
                    while (p < _s.Length && char.IsAsciiDigit(_s[p])) p++;
                }
                if (p < _s.Length && (_s[p] == 'e' || _s[p] == 'E'))
                {
                    int q = p + 1;
                    if (q < _s.Length && (_s[q] == '+' || _s[q] == '-')) q++;
                    if (q < _s.Length && char.IsAsciiDigit(_s[q]))
                    {
                        isFloat = true;
                        while (q < _s.Length && char.IsAsciiDigit(_s[q])) q++;
                        p = q;
                    }
                }
                string text = _s.Substring(start, p - start).Replace("_", "");
                if (isFloat) Add(TokenKind.Float, text, start, double.Parse(text, CultureInfo.InvariantCulture));
                else if (long.TryParse(text, NumberStyles.None, CultureInfo.InvariantCulture, out var lv)) Add(TokenKind.Int, text, start, lv);
                else Add(TokenKind.Float, text, start, double.Parse(text, CultureInfo.InvariantCulture));
            }
            else if (c == '"' || c == '\'')
            {
                p++;
                var sb = new StringBuilder();
                while (true)
                {
                    if (p >= _s.Length) throw Error("Unterminated string literal", start);
                    char ch = _s[p++];
                    if (ch == c) break;
                    if (ch != '\\')
                    {
                        sb.Append(ch);
                        continue;
                    }
                    if (p >= _s.Length) throw Error("Unterminated string literal", start);
                    char e = _s[p++];
                    switch (e)
                    {
                        case 'n': sb.Append('\n'); break;
                        case 't': sb.Append('\t'); break;
                        case 'r': sb.Append('\r'); break;
                        case 'b': sb.Append('\b'); break;
                        case 'f': sb.Append('\f'); break;
                        case 'v': sb.Append('\v'); break;
                        case 'a': sb.Append('\a'); break;
                        case '0': sb.Append('\0'); break;
                        case '\\': sb.Append('\\'); break;
                        case '\'': sb.Append('\''); break;
                        case '"': sb.Append('"'); break;
                        case '\n': break;
                        case 'x':
                            if (p + 2 <= _s.Length && int.TryParse(_s.AsSpan(p, 2), NumberStyles.HexNumber, CultureInfo.InvariantCulture, out var xv)) { sb.Append((char)xv); p += 2; }
                            else sb.Append("\\x");
                            break;
                        case 'u':
                            if (p + 4 <= _s.Length && int.TryParse(_s.AsSpan(p, 4), NumberStyles.HexNumber, CultureInfo.InvariantCulture, out var uv)) { sb.Append((char)uv); p += 4; }
                            else sb.Append("\\u");
                            break;
                        case 'U':
                            if (p + 8 <= _s.Length && int.TryParse(_s.AsSpan(p, 8), NumberStyles.HexNumber, CultureInfo.InvariantCulture, out var Uv) && Uv <= 0x10FFFF) { sb.Append(char.ConvertFromUtf32(Uv)); p += 8; }
                            else sb.Append("\\U");
                            break;
                        default: sb.Append('\\').Append(e); break;
                    }
                }
                string val = sb.ToString();
                Add(TokenKind.String, val, start, val);
            }
            else
            {
                string op;
                if (p + 1 < _s.Length && IsTwoCharOp(c, _s[p + 1])) op = _s.Substring(p, 2);
                else if ("+-*/%~<>=|.,:;()[]{}!".IndexOf(c) >= 0) op = c.ToString();
                else throw Error($"Unexpected character '{c}'", p);
                p += op.Length;
                if (op == "(" || op == "[" || op == "{") depth++;
                else if (op == ")" || op == "]" || op == "}") depth--;
                Add(TokenKind.Op, op, start);
            }
        }
    }

    private static bool IsTwoCharOp(char a, char b) =>
        (a == '/' && b == '/') || (a == '*' && b == '*') || (b == '=' && (a == '=' || a == '!' || a == '<' || a == '>'));
}
