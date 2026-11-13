using System.Globalization;
using System.Text;

namespace Sharplight.Templating;

/// <summary>JSON writing with the exact output format of Python's <c>json.dumps</c>, and a plain JSON reader.</summary>
public static class JinjaJson
{
    /// <summary>
    /// Serializes like <c>json.dumps(value, ensure_ascii=..., indent=..., separators=..., sort_keys=...)</c>.
    /// With <paramref name="indent"/> null the separators default to <c>", "</c> and <c>": "</c>; with an indent to <c>","</c> and <c>": "</c>.
    /// </summary>
    public static string Dumps(object? value, bool ensureAscii = false, string? indent = null, string? itemSeparator = null, string? keySeparator = null, bool sortKeys = false)
    {
        var sb = new StringBuilder();
        string item = itemSeparator ?? (indent != null ? "," : ", ");
        string key = keySeparator ?? ": ";
        Write(sb, value, ensureAscii, indent, item, key, sortKeys, 0);
        return sb.ToString();
    }

    private static void Newline(StringBuilder sb, string indent, int level)
    {
        sb.Append('\n');
        for (int i = 0; i < level; i++)
        {
            sb.Append(indent);
        }
    }

    private static void Write(StringBuilder sb, object? v, bool ascii, string? indent, string item, string key, bool sort, int level)
    {
        switch (v)
        {
            case null: sb.Append("null"); break;
            case Undefined u:
                if (u.Strict)
                {
                    throw u.Fail();
                }

                sb.Append("null");
                break;
            case bool b: sb.Append(b ? "true" : "false"); break;
            case long l: sb.Append(l.ToString(CultureInfo.InvariantCulture)); break;
            case double d:
                sb.Append(double.IsNaN(d) ? "NaN" : double.IsPositiveInfinity(d) ? "Infinity" : double.IsNegativeInfinity(d) ? "-Infinity" : PyFormat.FloatRepr(d));
                break;
            case string s: WriteString(sb, s, ascii); break;
            case Markup mk: WriteString(sb, mk.Value, ascii); break;
            case List<object?> list:
            {
                if (list.Count == 0)
                {
                    sb.Append("[]");
                    break;
                }
                sb.Append('[');
                for (int i = 0; i < list.Count; i++)
                {
                    if (i > 0)
                        {
                            sb.Append(item);
                        }

                        if (indent != null)
                        {
                            Newline(sb, indent, level + 1);
                        }

                        Write(sb, list[i], ascii, indent, item, key, sort, level + 1);
                }
                if (indent != null)
                    {
                        Newline(sb, indent, level);
                    }

                    sb.Append(']');
                break;
            }
            case JinjaDict dict:
                WriteObject(sb, dict.Select(kv => (kv.Key, kv.Value)).ToList(), ascii, indent, item, key, sort, level);
                break;
            case JinjaNamespace ns:
                WriteObject(sb, ns.Attributes.Select(kv => ((object)kv.Key, kv.Value)).ToList(), ascii, indent, item, key, sort, level);
                break;
            default:
                throw new TemplateRuntimeException($"Object of type {Ops.TypeName(v)} is not JSON serializable");
        }
    }

    private static void WriteObject(StringBuilder sb, List<(object Key, object? Value)> entries, bool ascii, string? indent, string item, string key, bool sort, int level)
    {
        if (entries.Count == 0)
        {
            sb.Append("{}");
            return;
        }
        var keyed = entries.Select(e => (Name: KeyString(e.Key), e.Value)).ToList();
        if (sort)
        {
            keyed = keyed.OrderBy(e => e.Name, StringComparer.Ordinal).ToList();
        }

        sb.Append('{');
        for (int i = 0; i < keyed.Count; i++)
        {
            if (i > 0)
            {
                sb.Append(item);
            }

            if (indent != null)
            {
                Newline(sb, indent, level + 1);
            }

            WriteString(sb, keyed[i].Name, ascii);
            sb.Append(key);
            Write(sb, keyed[i].Value, ascii, indent, item, key, sort, level + 1);
        }
        if (indent != null)
        {
            Newline(sb, indent, level);
        }

        sb.Append('}');
    }

    private static string KeyString(object k) => k switch
    {
        string s => s,
        bool b => b ? "true" : "false",
        long l => l.ToString(CultureInfo.InvariantCulture),
        double d => PyFormat.FloatRepr(d),
        _ => Ops.Str(k),
    };

    internal static void WriteString(StringBuilder sb, string s, bool ascii)
    {
        sb.Append('"');
        foreach (char c in s)
        {
            switch (c)
            {
                case '"': sb.Append("\\\""); break;
                case '\\': sb.Append("\\\\"); break;
                case '\n': sb.Append("\\n"); break;
                case '\r': sb.Append("\\r"); break;
                case '\t': sb.Append("\\t"); break;
                case '\b': sb.Append("\\b"); break;
                case '\f': sb.Append("\\f"); break;
                default:
                    if (c < 0x20 || (ascii && c > 0x7e))
                    {
                        sb.Append("\\u").Append(((int)c).ToString("x4", CultureInfo.InvariantCulture));
                    }
                    else
                    {
                        sb.Append(c);
                    }

                    break;
            }
        }
        sb.Append('"');
    }

    /// <summary>Parses JSON into the template value model (JinjaDict / List / string / long / double / bool / null).</summary>
    public static object? Parse(string json)
    {
        int i = 0;
        var v = ParseValue(json, ref i);
        SkipWs(json, ref i);
        if (i != json.Length)
        {
            throw new TemplateRuntimeException($"Extra data in JSON at position {i}");
        }

        return v;
    }

    private static void SkipWs(string s, ref int i)
    {
        while (i < s.Length && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r'))
        {
            i++;
        }
    }

    private static object? ParseValue(string s, ref int i)
    {
        SkipWs(s, ref i);
        if (i >= s.Length)
        {
            throw new TemplateRuntimeException("Unexpected end of JSON");
        }

        char c = s[i];
        switch (c)
        {
            case '{':
            {
                i++;
                var d = new JinjaDict();
                SkipWs(s, ref i);
                if (i < s.Length && s[i] == '}')
                {
                    i++;
                    return d;
                }
                while (true)
                {
                    SkipWs(s, ref i);
                    if (i >= s.Length || s[i] != '"')
                        {
                            throw new TemplateRuntimeException($"Expecting property name enclosed in double quotes at position {i}");
                        }

                        string k = ParseString(s, ref i);
                    SkipWs(s, ref i);
                    if (i >= s.Length || s[i] != ':')
                        {
                            throw new TemplateRuntimeException($"Expecting ':' delimiter at position {i}");
                        }

                        i++;
                    d[k] = ParseValue(s, ref i);
                    SkipWs(s, ref i);
                    if (i < s.Length && s[i] == ',') { i++; continue; }
                    if (i < s.Length && s[i] == '}') { i++; return d; }
                    throw new TemplateRuntimeException($"Expecting ',' delimiter at position {i}");
                }
            }
            case '[':
            {
                i++;
                var l = new List<object?>();
                SkipWs(s, ref i);
                if (i < s.Length && s[i] == ']')
                {
                    i++;
                    return l;
                }
                while (true)
                {
                    l.Add(ParseValue(s, ref i));
                    SkipWs(s, ref i);
                    if (i < s.Length && s[i] == ',') { i++; continue; }
                    if (i < s.Length && s[i] == ']') { i++; return l; }
                    throw new TemplateRuntimeException($"Expecting ',' delimiter at position {i}");
                }
            }
            case '"': return ParseString(s, ref i);
            case 't' when string.CompareOrdinal(s, i, "true", 0, 4) == 0: i += 4; return true;
            case 'f' when string.CompareOrdinal(s, i, "false", 0, 5) == 0: i += 5; return false;
            case 'n' when string.CompareOrdinal(s, i, "null", 0, 4) == 0: i += 4; return null;
            case 'N' when string.CompareOrdinal(s, i, "NaN", 0, 3) == 0: i += 3; return double.NaN;
            case 'I' when string.CompareOrdinal(s, i, "Infinity", 0, 8) == 0: i += 8; return double.PositiveInfinity;
            default:
            {
                int st = i;
                if (s[i] == '-')
                    {
                        i++;
                    }

                    if (string.CompareOrdinal(s, i, "Infinity", 0, 8) == 0) { i += 8; return double.NegativeInfinity; }
                bool isFloat = false;
                while (i < s.Length)
                {
                    char ch = s[i];
                    if (char.IsAsciiDigit(ch))
                        {
                            i++;
                        }
                        else if (ch == '.' || ch == 'e' || ch == 'E' || ((ch == '+' || ch == '-') && (s[i - 1] == 'e' || s[i - 1] == 'E')))
                    {
                        isFloat |= ch != '+' && ch != '-';
                        i++;
                    }
                    else
                        {
                            break;
                        }
                    }
                if (i == st)
                    {
                        throw new TemplateRuntimeException($"Expecting value at position {i}");
                    }

                    var span = s.AsSpan(st, i - st);
                if (!isFloat && long.TryParse(span, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out var lv))
                    {
                        return lv;
                    }

                    if (double.TryParse(span, NumberStyles.Float, CultureInfo.InvariantCulture, out var dv))
                    {
                        return dv;
                    }

                    throw new TemplateRuntimeException($"Expecting value at position {st}");
            }
        }
    }

    private static string ParseString(string s, ref int i)
    {
        i++; // opening quote
        var sb = new StringBuilder();
        while (true)
        {
            if (i >= s.Length)
            {
                throw new TemplateRuntimeException("Unterminated string in JSON");
            }

            char c = s[i++];
            if (c == '"')
            {
                return sb.ToString();
            }

            if (c != '\\')
            {
                sb.Append(c);
                continue;
            }
            if (i >= s.Length)
            {
                throw new TemplateRuntimeException("Unterminated string in JSON");
            }

            char e = s[i++];
            switch (e)
            {
                case 'n': sb.Append('\n'); break;
                case 't': sb.Append('\t'); break;
                case 'r': sb.Append('\r'); break;
                case 'b': sb.Append('\b'); break;
                case 'f': sb.Append('\f'); break;
                case 'u':
                    if (i + 4 > s.Length)
                    {
                        throw new TemplateRuntimeException("Invalid \\u escape in JSON");
                    }

                    sb.Append((char)int.Parse(s.AsSpan(i, 4), NumberStyles.HexNumber, CultureInfo.InvariantCulture));
                    i += 4;
                    break;
                default: sb.Append(e); break;
            }
        }
    }
}
