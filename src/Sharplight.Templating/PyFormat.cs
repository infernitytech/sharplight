using System.Globalization;
using System.Text;

namespace Sharplight.Templating;

/// <summary>Python-compatible text formatting helpers (float repr, str repr, % and str.format, strftime).</summary>
internal static class PyFormat
{
    public static string FloatRepr(double d)
    {
        if (double.IsNaN(d)) return "nan";
        if (double.IsPositiveInfinity(d)) return "inf";
        if (double.IsNegativeInfinity(d)) return "-inf";
        if (d == 0) return double.IsNegative(d) ? "-0.0" : "0.0";

        string r = d.ToString("R", CultureInfo.InvariantCulture);
        bool neg = false;
        if (r[0] == '-')
        {
            neg = true;
            r = r.Substring(1);
        }
        int exp = 0;
        int ei = r.IndexOfAny(new[] { 'E', 'e' });
        if (ei >= 0)
        {
            exp = int.Parse(r.AsSpan(ei + 1), NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture);
            r = r.Substring(0, ei);
        }
        int dot = r.IndexOf('.');
        string digits;
        int pointPos;
        if (dot >= 0)
        {
            digits = r.Substring(0, dot) + r.Substring(dot + 1);
            pointPos = dot;
        }
        else
        {
            digits = r;
            pointPos = r.Length;
        }
        pointPos += exp;
        int lead = 0;
        while (lead < digits.Length - 1 && digits[lead] == '0') lead++;
        digits = digits.Substring(lead);
        pointPos -= lead;
        digits = digits.TrimEnd('0');
        if (digits.Length == 0) digits = "0";

        int sci = pointPos - 1;
        var sb = new StringBuilder();
        if (neg) sb.Append('-');
        if (sci >= -4 && sci < 16)
        {
            if (pointPos <= 0)
            {
                sb.Append("0.").Append('0', -pointPos).Append(digits);
            }
            else if (pointPos >= digits.Length)
            {
                sb.Append(digits).Append('0', pointPos - digits.Length).Append(".0");
            }
            else
            {
                sb.Append(digits, 0, pointPos).Append('.').Append(digits, pointPos, digits.Length - pointPos);
            }
        }
        else
        {
            sb.Append(digits[0]);
            if (digits.Length > 1) sb.Append('.').Append(digits, 1, digits.Length - 1);
            sb.Append('e').Append(sci < 0 ? '-' : '+');
            int a = Math.Abs(sci);
            if (a < 10) sb.Append('0');
            sb.Append(a.ToString(CultureInfo.InvariantCulture));
        }
        return sb.ToString();
    }

    public static string StringRepr(string s)
    {
        char quote = s.Contains('\'') && !s.Contains('"') ? '"' : '\'';
        var sb = new StringBuilder();
        sb.Append(quote);
        foreach (char c in s)
        {
            if (c == quote || c == '\\') sb.Append('\\').Append(c);
            else if (c == '\n') sb.Append("\\n");
            else if (c == '\r') sb.Append("\\r");
            else if (c == '\t') sb.Append("\\t");
            else if (c < 0x20 || c == 0x7f) sb.Append("\\x").Append(((int)c).ToString("x2", CultureInfo.InvariantCulture));
            else sb.Append(c);
        }
        sb.Append(quote);
        return sb.ToString();
    }

    /// <summary>Python <c>fmt % args</c> (subset: s r d i f x o %, flags - 0 + space, width, precision).</summary>
    public static string Percent(string fmt, object? args)
    {
        var list = args is List<object?> l ? l : new List<object?> { args };
        var sb = new StringBuilder();
        int ai = 0;
        for (int i = 0; i < fmt.Length; i++)
        {
            char c = fmt[i];
            if (c != '%')
            {
                sb.Append(c);
                continue;
            }
            i++;
            if (i >= fmt.Length) throw new TemplateRuntimeException("incomplete format");
            bool left = false, zero = false, plus = false, space = false;
            for (; i < fmt.Length; i++)
            {
                if (fmt[i] == '-') left = true;
                else if (fmt[i] == '0') zero = true;
                else if (fmt[i] == '+') plus = true;
                else if (fmt[i] == ' ') space = true;
                else break;
            }
            int width = 0;
            while (i < fmt.Length && char.IsAsciiDigit(fmt[i])) width = width * 10 + (fmt[i++] - '0');
            int prec = -1;
            if (i < fmt.Length && fmt[i] == '.')
            {
                i++;
                prec = 0;
                while (i < fmt.Length && char.IsAsciiDigit(fmt[i])) prec = prec * 10 + (fmt[i++] - '0');
            }
            if (i >= fmt.Length) throw new TemplateRuntimeException("incomplete format");
            char conv = fmt[i];
            if (conv == '%')
            {
                sb.Append('%');
                continue;
            }
            if (ai >= list.Count) throw new TemplateRuntimeException("not enough arguments for format string");
            object? arg = list[ai++];
            string text;
            switch (conv)
            {
                case 's': text = Ops.Str(arg); if (prec >= 0 && text.Length > prec) text = text.Substring(0, prec); break;
                case 'r': text = Ops.Repr(arg); break;
                case 'd':
                case 'i':
                {
                    long v = arg is double dv ? (long)dv : Ops.ToLong(arg);
                    text = v.ToString(CultureInfo.InvariantCulture);
                    if (v >= 0) text = (plus ? "+" : space ? " " : "") + text;
                    break;
                }
                case 'f':
                case 'F':
                {
                    double v = Ops.ToDouble(arg);
                    text = v.ToString("F" + (prec < 0 ? 6 : prec), CultureInfo.InvariantCulture);
                    if (v >= 0) text = (plus ? "+" : space ? " " : "") + text;
                    break;
                }
                case 'x': text = Ops.ToLong(arg).ToString("x", CultureInfo.InvariantCulture); break;
                case 'X': text = Ops.ToLong(arg).ToString("X", CultureInfo.InvariantCulture); break;
                case 'o': text = Convert.ToString(Ops.ToLong(arg), 8); break;
                default: throw new TemplateRuntimeException($"unsupported format character '{conv}'");
            }
            if (text.Length < width)
            {
                if (left) text = text.PadRight(width);
                else if (zero && conv != 's' && conv != 'r')
                {
                    bool sign = text.Length > 0 && (text[0] == '-' || text[0] == '+');
                    text = (sign ? text.Substring(0, 1) : "") + new string('0', width - text.Length) + (sign ? text.Substring(1) : text);
                }
                else text = text.PadLeft(width);
            }
            sb.Append(text);
        }
        return sb.ToString();
    }

    /// <summary>Python <c>str.format</c> (subset: {} {0} {name} {a.b} {a[0]} !r !s and a simple width/fill/precision spec).</summary>
    public static string StrFormat(string fmt, CallArgs args)
    {
        var sb = new StringBuilder();
        int auto = 0;
        for (int i = 0; i < fmt.Length; i++)
        {
            char c = fmt[i];
            if (c == '{')
            {
                if (i + 1 < fmt.Length && fmt[i + 1] == '{')
                {
                    sb.Append('{');
                    i++;
                    continue;
                }
                int end = fmt.IndexOf('}', i);
                if (end < 0) throw new TemplateRuntimeException("Single '{' encountered in format string");
                string field = fmt.Substring(i + 1, end - i - 1);
                i = end;
                string spec = "";
                char conv = '\0';
                int ci = field.IndexOf(':');
                if (ci >= 0)
                {
                    spec = field.Substring(ci + 1);
                    field = field.Substring(0, ci);
                }
                int bi = field.IndexOf('!');
                if (bi >= 0)
                {
                    conv = bi + 1 < field.Length ? field[bi + 1] : '\0';
                    field = field.Substring(0, bi);
                }
                // split first name / index and the accessor chain
                int k = 0;
                while (k < field.Length && field[k] != '.' && field[k] != '[') k++;
                string first = field.Substring(0, k);
                object? val;
                if (first.Length == 0) val = args.Positional.Count > auto ? args.Positional[auto++] : throw new TemplateRuntimeException("Replacement index out of range for positional args tuple");
                else if (char.IsAsciiDigit(first[0]))
                {
                    int idx = int.Parse(first, CultureInfo.InvariantCulture);
                    val = idx < args.Positional.Count ? args.Positional[idx] : throw new TemplateRuntimeException("Replacement index out of range for positional args tuple");
                }
                else val = args.Named != null && args.Named.TryGetValue(first, out var nv) ? nv : throw new TemplateRuntimeException($"KeyError: '{first}'");
                while (k < field.Length)
                {
                    if (field[k] == '.')
                    {
                        int s = ++k;
                        while (k < field.Length && field[k] != '.' && field[k] != '[') k++;
                        val = Ops.GetAttr(val, field.Substring(s, k - s), false);
                    }
                    else
                    {
                        int e = field.IndexOf(']', k);
                        string key = field.Substring(k + 1, e - k - 1);
                        val = Ops.GetItem(val, long.TryParse(key, out var li) ? li : key, false);
                        k = e + 1;
                    }
                }
                string text = conv == 'r' ? Ops.Repr(val) : Ops.Str(val);
                sb.Append(ApplySpec(text, val, spec));
            }
            else if (c == '}')
            {
                if (i + 1 < fmt.Length && fmt[i + 1] == '}') i++;
                else throw new TemplateRuntimeException("Single '}' encountered in format string");
                sb.Append('}');
            }
            else sb.Append(c);
        }
        return sb.ToString();
    }

    private static string ApplySpec(string text, object? val, string spec)
    {
        if (spec.Length == 0) return text;
        int i = 0;
        char fill = ' ', align = '\0';
        if (spec.Length >= 2 && "<>^".IndexOf(spec[1]) >= 0)
        {
            fill = spec[0];
            align = spec[1];
            i = 2;
        }
        else if (spec.Length >= 1 && "<>^".IndexOf(spec[0]) >= 0)
        {
            align = spec[0];
            i = 1;
        }
        bool zero = false;
        if (i < spec.Length && spec[i] == '0')
        {
            zero = true;
            i++;
        }
        int width = 0;
        while (i < spec.Length && char.IsAsciiDigit(spec[i])) width = width * 10 + (spec[i++] - '0');
        int prec = -1;
        if (i < spec.Length && spec[i] == '.')
        {
            i++;
            prec = 0;
            while (i < spec.Length && char.IsAsciiDigit(spec[i])) prec = prec * 10 + (spec[i++] - '0');
        }
        char type = i < spec.Length ? spec[i] : '\0';
        if (type == 'f' && Ops.IsNum(val)) text = Ops.ToDouble(val).ToString("F" + (prec < 0 ? 6 : prec), CultureInfo.InvariantCulture);
        else if (type == 'd' && Ops.IsNum(val)) text = Ops.ToLong(val).ToString(CultureInfo.InvariantCulture);
        else if (prec >= 0 && val is string && text.Length > prec) text = text.Substring(0, prec);
        if (text.Length >= width) return text;
        if (align == '\0')
        {
            if (zero) { align = '>'; fill = '0'; }
            else align = Ops.IsNum(val) ? '>' : '<';
        }
        int pad = width - text.Length;
        return align switch
        {
            '<' => text + new string(fill, pad),
            '>' => new string(fill, pad) + text,
            _ => new string(fill, pad / 2) + text + new string(fill, pad - pad / 2),
        };
    }

    /// <summary>Python <c>time.strftime</c> subset.</summary>
    public static string Strftime(string fmt, DateTimeOffset t)
    {
        var sb = new StringBuilder();
        var ci = CultureInfo.InvariantCulture;
        for (int i = 0; i < fmt.Length; i++)
        {
            char c = fmt[i];
            if (c != '%' || i + 1 >= fmt.Length)
            {
                sb.Append(c);
                continue;
            }
            char f = fmt[++i];
            bool noPad = false;
            if (f == '-' && i + 1 < fmt.Length)
            {
                noPad = true;
                f = fmt[++i];
            }
            string Num(int v, int w) => noPad ? v.ToString(ci) : v.ToString(ci).PadLeft(w, '0');
            switch (f)
            {
                case 'Y': sb.Append(t.Year.ToString(ci)); break;
                case 'y': sb.Append(Num(t.Year % 100, 2)); break;
                case 'm': sb.Append(Num(t.Month, 2)); break;
                case 'd': sb.Append(Num(t.Day, 2)); break;
                case 'e': sb.Append(t.Day.ToString(ci).PadLeft(2)); break;
                case 'H': sb.Append(Num(t.Hour, 2)); break;
                case 'I': sb.Append(Num(t.Hour % 12 == 0 ? 12 : t.Hour % 12, 2)); break;
                case 'M': sb.Append(Num(t.Minute, 2)); break;
                case 'S': sb.Append(Num(t.Second, 2)); break;
                case 'f': sb.Append((t.Ticks % 10_000_000 / 10).ToString(ci).PadLeft(6, '0')); break;
                case 'p': sb.Append(t.Hour < 12 ? "AM" : "PM"); break;
                case 'B': sb.Append(t.ToString("MMMM", ci)); break;
                case 'b': sb.Append(t.ToString("MMM", ci)); break;
                case 'A': sb.Append(t.ToString("dddd", ci)); break;
                case 'a': sb.Append(t.ToString("ddd", ci)); break;
                case 'j': sb.Append(Num(t.DayOfYear, 3)); break;
                case 'w': sb.Append(((int)t.DayOfWeek).ToString(ci)); break;
                case 'Z': sb.Append(t.Offset == TimeSpan.Zero ? "UTC" : ""); break;
                case 'z': sb.Append((t.Offset < TimeSpan.Zero ? "-" : "+") + Math.Abs(t.Offset.Hours).ToString("00", ci) + Math.Abs(t.Offset.Minutes).ToString("00", ci)); break;
                case 'F': sb.Append(t.ToString("yyyy-MM-dd", ci)); break;
                case 'T': sb.Append(t.ToString("HH:mm:ss", ci)); break;
                case 'D': sb.Append(t.ToString("MM/dd/yy", ci)); break;
                case 'c': sb.Append(t.ToString("ddd MMM ", ci)).Append(t.Day.ToString(ci).PadLeft(2)).Append(t.ToString(" HH:mm:ss yyyy", ci)); break;
                case 'x': sb.Append(t.ToString("MM/dd/yy", ci)); break;
                case 'X': sb.Append(t.ToString("HH:mm:ss", ci)); break;
                case '%': sb.Append('%'); break;
                default: sb.Append('%').Append(f); break;
            }
        }
        return sb.ToString();
    }
}
