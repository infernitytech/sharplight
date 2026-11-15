using System.Globalization;
using System.Text;

namespace Sharplight.Templating;

/// <summary>Semantics of the dynamic value model (Python-like).</summary>
internal static class Ops
{
    public static bool IsNum(object? o) => o is long or double or bool;
    public static bool IsInt(object? o) => o is long or bool;
    public static long ToLong(object? o) => o switch { bool b => b ? 1 : 0, long l => l, Undefined { Strict: true } u => throw u.Fail(), _ => throw new TemplateRuntimeException($"expected an integer, got '{TypeName(o)}'") };
    public static double ToDouble(object? o) => o switch { bool b => b ? 1.0 : 0.0, long l => l, double d => d, Undefined { Strict: true } u => throw u.Fail(), _ => throw new TemplateRuntimeException($"expected a number, got '{TypeName(o)}'") };

    public static Undefined Undef(string? name, bool strict) => new(name, strict);

    public static string TypeName(object? o) => o switch
    {
        null => "NoneType",
        bool => "bool",
        long => "int",
        double => "float",
        string => "str",
        Markup => "Markup",
        List<object?> => "list",
        JinjaDict => "dict",
        JinjaNamespace => "Namespace",
        JinjaCallable => "function",
        Undefined => "Undefined",
        LoopContext => "LoopContext",
        _ => o.GetType().Name,
    };

    public static bool Truthy(object? o)
    {
        switch (o)
        {
            case null: return false;
            case bool b: return b;
            case long l: return l != 0;
            case double d: return d != 0;
            case string s: return s.Length > 0;
            case Markup mk: return mk.Value.Length > 0;
            case List<object?> li: return li.Count > 0;
            case JinjaDict d: return d.Count > 0;
            case Undefined u:
                if (u.Strict)
                {
                    throw u.Fail();
                }

                return false;
            default: return true;
        }
    }

    // ---------------------------------------------------------------- string conversion

    public static string Str(object? o)
    {
        switch (o)
        {
            case string s: return s;
            case Markup mk: return mk.Value;
            case null: return "None";
            case bool b: return b ? "True" : "False";
            case long l: return l.ToString(CultureInfo.InvariantCulture);
            case double d: return PyFormat.FloatRepr(d);
            case Undefined u:
                if (u.Strict)
                {
                    throw u.Fail();
                }

                return "";
            default: return Repr(o);
        }
    }

    public static string Repr(object? o)
    {
        switch (o)
        {
            case string s: return PyFormat.StringRepr(s);
            case Markup mk: return "Markup(" + PyFormat.StringRepr(mk.Value) + ")";
            case List<object?> l:
            {
                bool tuple = l is JinjaTuple;
                var sb = new StringBuilder(tuple ? "(" : "[");
                for (int i = 0; i < l.Count; i++)
                {
                    if (i > 0)
                        {
                            sb.Append(", ");
                        }

                        sb.Append(Repr(l[i]));
                }
                if (tuple && l.Count == 1)
                    {
                        sb.Append(',');
                    }

                    return sb.Append(tuple ? ')' : ']').ToString();
            }
            case JinjaDict d:
            {
                var sb = new StringBuilder("{");
                bool first = true;
                foreach (var kv in d)
                {
                    if (!first)
                        {
                            sb.Append(", ");
                        }

                        first = false;
                    sb.Append(Repr(kv.Key)).Append(": ").Append(Repr(kv.Value));
                }
                return sb.Append('}').ToString();
            }
            case JinjaNamespace ns:
            {
                var sb = new StringBuilder("<Namespace");
                foreach (var kv in ns.Attributes)
                    {
                        sb.Append(' ').Append(kv.Key).Append('=').Append(Repr(kv.Value));
                    }

                    return sb.Append('>').ToString();
            }
            case JinjaCallable c: return $"<function {c.Name}>";
            case LoopContext: return "<LoopContext>";
            case Undefined: return "Undefined";
            default: return Str(o);
        }
    }

    // ---------------------------------------------------------------- comparison

    public static bool Eq(object? a, object? b)
    {
        if (a is Markup ma)
        {
            a = ma.Value;
        }

        if (b is Markup mb)
        {
            b = mb.Value;
        }

        if (a is Undefined)
        {
            return b is Undefined;
        }

        if (b is Undefined)
        {
            return false;
        }

        if (a is null)
        {
            return b is null;
        }

        if (b is null)
        {
            return false;
        }

        if (IsNum(a) && IsNum(b))
        {
            if (IsInt(a) && IsInt(b))
            {
                return ToLong(a) == ToLong(b);
            }

            return ToDouble(a) == ToDouble(b);
        }
        switch (a)
        {
            case string s: return b is string t && string.Equals(s, t, StringComparison.Ordinal);
            case List<object?> la:
            {
                if (b is not List<object?> lb || la.Count != lb.Count)
                    {
                        return false;
                    }

                    for (int i = 0; i < la.Count; i++)
                    {
                        if (!Eq(la[i], lb[i]))
                        {
                            return false;
                        }
                    }

                    return true;
            }
            case JinjaDict da:
            {
                if (b is not JinjaDict db || da.Count != db.Count)
                    {
                        return false;
                    }

                    foreach (var kv in da)
                {
                    if (!db.TryGetValue(kv.Key, out var ov) || !Eq(kv.Value, ov))
                        {
                            return false;
                        }
                    }
                return true;
            }
            default: return ReferenceEquals(a, b);
        }
    }

    public static int Compare(object? a, object? b, string op = "<")
    {
        if (a is Markup ma)
        {
            a = ma.Value;
        }

        if (b is Markup mb)
        {
            b = mb.Value;
        }

        if (IsNum(a) && IsNum(b))
        {
            if (IsInt(a) && IsInt(b))
            {
                return ToLong(a).CompareTo(ToLong(b));
            }

            return ToDouble(a).CompareTo(ToDouble(b));
        }
        if (a is string s && b is string t)
        {
            return Math.Sign(string.CompareOrdinal(s, t));
        }

        if (a is List<object?> la && b is List<object?> lb)
        {
            int n = Math.Min(la.Count, lb.Count);
            for (int i = 0; i < n; i++)
            {
                if (Eq(la[i], lb[i]))
                {
                    continue;
                }

                return Compare(la[i], lb[i], op);
            }
            return la.Count.CompareTo(lb.Count);
        }
        if (a is Undefined ua && ua.Strict)
        {
            throw ua.Fail();
        }

        if (b is Undefined ub && ub.Strict)
        {
            throw ub.Fail();
        }

        throw new TemplateRuntimeException($"'{op}' not supported between instances of '{TypeName(a)}' and '{TypeName(b)}'");
    }

    public static bool Contains(object? container, object? item)
    {
        if (container is Markup mc)
        {
            container = mc.Value;
        }

        if (item is Markup mi)
        {
            item = mi.Value;
        }

        switch (container)
        {
            case string s:
                if (item is string t)
                {
                    return s.Contains(t, StringComparison.Ordinal);
                }

                throw new TemplateRuntimeException($"'in <string>' requires string as left operand, not {TypeName(item)}");
            case List<object?> l:
                foreach (var x in l)
                {
                    if (Eq(x, item))
                    {
                        return true;
                    }
                }

                return false;
            case JinjaDict d:
                return item is string or long or bool ? d.ContainsKey(item) : item is double dd && dd == Math.Floor(dd) && d.ContainsKey((long)dd);
            case JinjaNamespace ns:
                return item is string k && ns.Attributes.ContainsKey(k);
            case Undefined u:
                if (u.Strict)
                {
                    throw u.Fail();
                }

                return false;
            case null:
                throw new TemplateRuntimeException("argument of type 'NoneType' is not iterable");
            default:
                return false;
        }
    }

    // ---------------------------------------------------------------- arithmetic

    private static TemplateRuntimeException BinFail(string op, object? a, object? b)
    {
        if (a is Undefined ua && ua.Strict)
        {
            return ua.Fail();
        }

        if (b is Undefined ub && ub.Strict)
        {
            return ub.Fail();
        }

        return new TemplateRuntimeException($"unsupported operand type(s) for {op}: '{TypeName(a)}' and '{TypeName(b)}'");
    }

    private static string MarkupText(object? v) => v is Markup m ? m.Value : Markup.Escape((string)v!);

    public static object? Add(object? a, object? b)
    {
        if (IsNum(a) && IsNum(b))
        {
            return IsInt(a) && IsInt(b) ? (object)(ToLong(a) + ToLong(b)) : ToDouble(a) + ToDouble(b);
        }

        if (a is string s && b is string t)
        {
            return string.Concat(s, t);
        }

        if ((a is Markup || b is Markup) && (a is string or Markup) && (b is string or Markup))
        {
            return new Markup(MarkupText(a) + MarkupText(b));
        }

        if (a is List<object?> la && b is List<object?> lb)
        {
            var r = new List<object?>(la.Count + lb.Count);
            r.AddRange(la);
            r.AddRange(lb);
            return r;
        }
        throw BinFail("+", a, b);
    }

    public static object? Sub(object? a, object? b)
    {
        if (IsNum(a) && IsNum(b))
        {
            return IsInt(a) && IsInt(b) ? (object)(ToLong(a) - ToLong(b)) : ToDouble(a) - ToDouble(b);
        }

        throw BinFail("-", a, b);
    }

    public static object? Mul(object? a, object? b)
    {
        if (IsNum(a) && IsNum(b))
        {
            return IsInt(a) && IsInt(b) ? (object)(ToLong(a) * ToLong(b)) : ToDouble(a) * ToDouble(b);
        }

        if (a is string s && IsInt(b))
        {
            return Repeat(s, ToLong(b));
        }

        if (b is string t && IsInt(a))
        {
            return Repeat(t, ToLong(a));
        }

        if (a is List<object?> l && IsInt(b))
        {
            var r = new List<object?>();
            for (long i = 0; i < ToLong(b); i++)
            {
                r.AddRange(l);
            }

            return r;
        }
        throw BinFail("*", a, b);
    }

    private static string Repeat(string s, long n)
    {
        if (n <= 0)
        {
            return "";
        }

        var sb = new StringBuilder(s.Length * (int)n);
        for (long i = 0; i < n; i++)
        {
            sb.Append(s);
        }

        return sb.ToString();
    }

    public static object? Div(object? a, object? b)
    {
        if (IsNum(a) && IsNum(b))
        {
            double d = ToDouble(b);
            if (d == 0)
            {
                throw new TemplateRuntimeException("division by zero");
            }

            return ToDouble(a) / d;
        }
        throw BinFail("/", a, b);
    }

    public static object? FloorDiv(object? a, object? b)
    {
        if (IsNum(a) && IsNum(b))
        {
            if (IsInt(a) && IsInt(b))
            {
                long x = ToLong(a);
                long y = ToLong(b);
                if (y == 0)
                {
                    throw new TemplateRuntimeException("integer division or modulo by zero");
                }

                long q = x / y;
                if ((x % y != 0) && ((x < 0) != (y < 0)))
                {
                    q--;
                }

                return q;
            }
            double dy = ToDouble(b);
            if (dy == 0)
            {
                throw new TemplateRuntimeException("float floor division by zero");
            }

            return Math.Floor(ToDouble(a) / dy);
        }
        throw BinFail("//", a, b);
    }

    public static object? Mod(object? a, object? b)
    {
        if (IsNum(a) && IsNum(b))
        {
            if (IsInt(a) && IsInt(b))
            {
                long x = ToLong(a);
                long y = ToLong(b);
                if (y == 0)
                {
                    throw new TemplateRuntimeException("integer division or modulo by zero");
                }

                long r = x % y;
                if (r != 0 && ((r < 0) != (y < 0)))
                {
                    r += y;
                }

                return r;
            }
            double dx = ToDouble(a), dy = ToDouble(b);
            if (dy == 0)
            {
                throw new TemplateRuntimeException("float modulo by zero");
            }

            double m = dx % dy;
            if (m != 0 && ((m < 0) != (dy < 0)))
            {
                m += dy;
            }

            return m;
        }
        if (a is string fmt)
        {
            return PyFormat.Percent(fmt, b);
        }

        throw BinFail("%", a, b);
    }

    public static object? Pow(object? a, object? b)
    {
        if (IsNum(a) && IsNum(b))
        {
            if (IsInt(a) && IsInt(b) && ToLong(b) >= 0)
            {
                long x = ToLong(a), e = ToLong(b), r = 1;
                try
                {
                    checked
                    {
                        for (long i = 0; i < e; i++)
                        {
                            r *= x;
                        }
                    }
                    return r;
                }
                catch (OverflowException)
                {
                    return Math.Pow(x, e);
                }
            }
            return Math.Pow(ToDouble(a), ToDouble(b));
        }
        throw BinFail("**", a, b);
    }

    public static object? Neg(object? a)
    {
        if (a is double d)
        {
            return -d;
        }

        if (IsInt(a))
        {
            return -ToLong(a);
        }

        if (a is Undefined u && u.Strict)
        {
            throw u.Fail();
        }

        throw new TemplateRuntimeException($"bad operand type for unary -: '{TypeName(a)}'");
    }

    public static object? Pos(object? a)
    {
        if (a is double)
        {
            return a;
        }

        if (IsInt(a))
        {
            return ToLong(a);
        }

        if (a is Undefined u && u.Strict)
        {
            throw u.Fail();
        }

        throw new TemplateRuntimeException($"bad operand type for unary +: '{TypeName(a)}'");
    }

    // ---------------------------------------------------------------- iteration & access

    public static List<object?> Iterate(object? o)
    {
        if (o is Markup mk)
        {
            o = mk.Value;
        }

        switch (o)
        {
            case List<object?> l: return l;
            case JinjaDict d: return new List<object?>(d.Keys);
            case string s:
            {
                var r = new List<object?>(s.Length);
                foreach (var rune in s.EnumerateRunes())
                    {
                        r.Add(rune.ToString());
                    }

                    return r;
            }
            case Undefined u:
                if (u.Strict)
                {
                    throw u.Fail();
                }

                return new List<object?>();
            default:
                throw new TemplateRuntimeException($"'{TypeName(o)}' object is not iterable");
        }
    }

    public static bool IsIterable(object? o) => o is List<object?> or JinjaDict or string or Markup;

    public static int CodePointCount(string s)
    {
        int n = s.Length;
        foreach (char c in s)
        {
            if (char.IsLowSurrogate(c))
            {
                n--;
            }
        }

        return n;
    }

    private static bool HasSurrogates(string s)
    {
        foreach (char c in s)
        {
            if (char.IsSurrogate(c))
            {
                return true;
            }
        }

        return false;
    }

    private static string[] CodePoints(string s)
    {
        var r = new List<string>(s.Length);
        foreach (var rune in s.EnumerateRunes())
        {
            r.Add(rune.ToString());
        }

        return r.ToArray();
    }

    public static int Len(object? o)
    {
        switch (o)
        {
            case string s: return CodePointCount(s);
            case Markup mk: return CodePointCount(mk.Value);
            case List<object?> l: return l.Count;
            case JinjaDict d: return d.Count;
            case Undefined u when u.Strict: throw u.Fail();
            case Undefined: return 0;
            case LoopContext lc: return lc.Length;
            default: throw new TemplateRuntimeException($"object of type '{TypeName(o)}' has no len()");
        }
    }

    public static object? GetAttr(object? o, string name, bool strict, bool forCall = false)
    {
        if (o is Markup mk)
        {
            o = mk.Value;
        }

        switch (o)
        {
            case JinjaDict d:
                if (forCall && Methods.HasDictMethod(name))
                {
                    return new BoundMethod(d, name);
                }

                if (d.TryGetValue(name, out var v))
                {
                    return v;
                }

                if (Methods.HasDictMethod(name))
                {
                    return new BoundMethod(d, name);
                }

                return new Undefined(null, strict);
            case JinjaNamespace ns:
                return ns.Attributes.TryGetValue(name, out var nv) ? nv : new Undefined(null, strict);
            case LoopContext lc:
                return lc.Get(name, strict);
            case string s:
                return Methods.HasStringMethod(name) ? new BoundMethod(s, name) : new Undefined(null, strict);
            case List<object?> l:
                return Methods.HasListMethod(name) ? new BoundMethod(l, name) : new Undefined(null, strict);
            case Undefined u:
                if (u.Strict)
                {
                    throw u.Fail();
                }

                return u;
            default:
                return new Undefined(null, strict);
        }
    }

    public static object? GetItem(object? o, object? key, bool strict)
    {
        if (o is Markup mk)
        {
            o = mk.Value;
        }

        switch (o)
        {
            case List<object?> l:
                if (key is SliceValue sl)
                {
                    return SliceList(l, sl);
                }

                if (IsInt(key))
                {
                    long i = ToLong(key);
                    if (i < 0)
                    {
                        i += l.Count;
                    }

                    return i >= 0 && i < l.Count ? l[(int)i] : new Undefined(null, strict);
                }
                return new Undefined(null, strict);
            case string s:
                if (key is SliceValue ss)
                {
                    return SliceString(s, ss);
                }

                if (IsInt(key))
                {
                    long i = ToLong(key);
                    if (HasSurrogates(s))
                    {
                        var cps = CodePoints(s);
                        if (i < 0)
                        {
                            i += cps.Length;
                        }

                        return i >= 0 && i < cps.Length ? cps[(int)i] : new Undefined(null, strict);
                    }
                    if (i < 0)
                    {
                        i += s.Length;
                    }

                    return i >= 0 && i < s.Length ? s[(int)i].ToString() : new Undefined(null, strict);
                }
                return new Undefined(null, strict);
            case JinjaDict d:
                if (key is string or long or bool)
                {
                    if (d.TryGetValue(key, out var v))
                    {
                        return v;
                    }

                    if (key is string ks && Methods.HasDictMethod(ks))
                    {
                        return new BoundMethod(d, ks);
                    }
                }
                else if (key is double dk && dk == Math.Floor(dk) && d.TryGetValue((long)dk, out var v2))
                {
                    return v2;
                }

                return new Undefined(null, strict);
            case JinjaNamespace ns:
                return key is string nk && ns.Attributes.TryGetValue(nk, out var nv) ? nv : new Undefined(null, strict);
            case LoopContext lc:
                return key is string lk ? lc.Get(lk, strict) : new Undefined(null, strict);
            case Undefined u:
                if (u.Strict)
                {
                    throw u.Fail();
                }

                return u;
            default:
                return new Undefined(null, strict);
        }
    }

    private static List<int> SliceIndices(int len, SliceValue sl)
    {
        long step = sl.Step is null or Undefined ? 1 : ToLong(sl.Step);
        if (step == 0)
        {
            throw new TemplateRuntimeException("slice step cannot be zero");
        }

        long start, stop;
        if (step > 0)
        {
            start = sl.Start is null or Undefined ? 0 : ToLong(sl.Start);
            stop = sl.Stop is null or Undefined ? len : ToLong(sl.Stop);
            if (start < 0)
            {
                start = Math.Max(0, start + len);
            }
            else
            {
                start = Math.Min(start, len);
            }

            if (stop < 0)
            {
                stop = Math.Max(0, stop + len);
            }
            else
            {
                stop = Math.Min(stop, len);
            }
        }
        else
        {
            start = sl.Start is null or Undefined ? len - 1 : ToLong(sl.Start);
            stop = sl.Stop is null or Undefined ? -1 : ToLong(sl.Stop);
            bool noStop = sl.Stop is null or Undefined;
            if (start < 0)
            {
                start = Math.Max(-1, start + len);
            }
            else
            {
                start = Math.Min(start, len - 1);
            }

            if (!noStop)
            {
                if (stop < 0)
                {
                    stop = Math.Max(-1, stop + len);
                }
                else
                {
                    stop = Math.Min(stop, len - 1);
                }
            }
        }
        var r = new List<int>();
        if (step > 0)
        {
            for (long i = start; i < stop; i += step)
            {
                r.Add((int)i);
            }
        }
        else
        {
            for (long i = start; i > stop; i += step)
            {
                r.Add((int)i);
            }
        }

        return r;
    }

    public static List<object?> SliceList(List<object?> l, SliceValue sl)
    {
        var r = new List<object?>();
        foreach (var i in SliceIndices(l.Count, sl))
        {
            r.Add(l[i]);
        }

        return r;
    }

    public static string SliceString(string s, SliceValue sl)
    {
        var sb = new StringBuilder();
        if (HasSurrogates(s))
        {
            var cps = CodePoints(s);
            foreach (var i in SliceIndices(cps.Length, sl))
            {
                sb.Append(cps[i]);
            }

            return sb.ToString();
        }
        foreach (var i in SliceIndices(s.Length, sl))
        {
            sb.Append(s[i]);
        }

        return sb.ToString();
    }
}
