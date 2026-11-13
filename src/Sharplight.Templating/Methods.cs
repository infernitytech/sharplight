using System.Text;

namespace Sharplight.Templating;

/// <summary>Python-style methods callable on strings, lists and dicts (<c>"x".strip()</c>, <c>d.items()</c>, <c>l.append(v)</c>).</summary>
internal static class Methods
{
    private static readonly HashSet<string> StringMethods = new()
    {
        "strip", "lstrip", "rstrip", "split", "rsplit", "splitlines", "startswith", "endswith", "upper", "lower",
        "capitalize", "title", "swapcase", "replace", "find", "rfind", "index", "rindex", "count", "join", "format",
        "isdigit", "isnumeric", "isdecimal", "isalpha", "isalnum", "isspace", "islower", "isupper", "zfill", "ljust",
        "rjust", "center", "removeprefix", "removesuffix", "partition", "rpartition",
    };

    private static readonly HashSet<string> ListMethods = new()
    {
        "append", "extend", "pop", "insert", "index", "count", "copy", "reverse", "clear", "remove",
    };

    private static readonly HashSet<string> DictMethods = new()
    {
        "items", "keys", "values", "get", "pop", "update", "setdefault", "copy", "clear",
    };

    public static bool HasStringMethod(string n) => StringMethods.Contains(n);
    public static bool HasListMethod(string n) => ListMethods.Contains(n);
    public static bool HasDictMethod(string n) => DictMethods.Contains(n);

    private static string Arg(CallArgs a, int i, string name, string method)
    {
        var v = a.Get(i, name);
        if (v is string s)
        {
            return s;
        }

        if (v is Markup mk)
        {
            return mk.Value;
        }

        throw new TemplateRuntimeException($"{method}() argument '{name}' must be str, not {Ops.TypeName(v)}");
    }

    private static TemplateRuntimeException NoMethod(object target, string name) =>
        new($"'{Ops.TypeName(target)}' object has no method '{name}'");

    public static object? Call(object target, string name, CallArgs a)
    {
        switch (target)
        {
            case Markup mk: return CallString(mk.Value, name, a);
            case string s: return CallString(s, name, a);
            case List<object?> l: return CallList(l, name, a);
            case JinjaDict d: return CallDict(d, name, a);
            case LoopContext lc: return lc.Call(name, a);
            default: throw NoMethod(target, name);
        }
    }

    // ------------------------------------------------------------------ strings

    internal static string PyStrip(string s, string? chars, bool left, bool right)
    {
        int start = 0, end = s.Length;
        bool Strip(char c) => chars == null ? char.IsWhiteSpace(c) : chars.IndexOf(c) >= 0;
        if (left)
        {
            while (start < end && Strip(s[start]))
            {
                start++;
            }
        }

        if (right)
        {
            while (end > start && Strip(s[end - 1]))
            {
                end--;
            }
        }

        return s.Substring(start, end - start);
    }

    internal static List<object?> PySplit(string s, string? sep, long maxsplit)
    {
        var r = new List<object?>();
        if (sep == null)
        {
            int i = 0, n = s.Length;
            long splits = 0;
            while (true)
            {
                while (i < n && char.IsWhiteSpace(s[i]))
                {
                    i++;
                }

                if (i >= n)
                {
                    break;
                }

                if (maxsplit >= 0 && splits >= maxsplit)
                {
                    r.Add(s.Substring(i).TrimEnd());
                    break;
                }
                int st = i;
                while (i < n && !char.IsWhiteSpace(s[i]))
                {
                    i++;
                }

                r.Add(s.Substring(st, i - st));
                splits++;
            }
            return r;
        }
        if (sep.Length == 0)
        {
            throw new TemplateRuntimeException("empty separator");
        }

        int pos = 0;
        long cnt = 0;
        while (true)
        {
            int idx = maxsplit >= 0 && cnt >= maxsplit ? -1 : s.IndexOf(sep, pos, StringComparison.Ordinal);
            if (idx < 0)
            {
                r.Add(s.Substring(pos));
                break;
            }
            r.Add(s.Substring(pos, idx - pos));
            pos = idx + sep.Length;
            cnt++;
        }
        return r;
    }

    internal static List<object?> PyRSplit(string s, string? sep, long maxsplit)
    {
        if (maxsplit < 0)
        {
            return PySplit(s, sep, -1);
        }

        var r = new List<object?>();
        if (sep == null)
        {
            int end = s.Length;
            long cnt = 0;
            while (true)
            {
                while (end > 0 && char.IsWhiteSpace(s[end - 1]))
                {
                    end--;
                }

                if (end <= 0)
                {
                    break;
                }

                if (cnt >= maxsplit)
                {
                    r.Add(s.Substring(0, end).TrimStart());
                    break;
                }
                int st = end;
                while (st > 0 && !char.IsWhiteSpace(s[st - 1]))
                {
                    st--;
                }

                r.Add(s.Substring(st, end - st));
                end = st;
                cnt++;
            }
            r.Reverse();
            return r;
        }
        if (sep.Length == 0)
        {
            throw new TemplateRuntimeException("empty separator");
        }

        int e = s.Length;
        long c = 0;
        while (true)
        {
            int idx = c >= maxsplit || e < sep.Length ? -1 : s.LastIndexOf(sep, e - 1, e, StringComparison.Ordinal);
            if (idx < 0)
            {
                r.Add(s.Substring(0, e));
                break;
            }
            r.Add(s.Substring(idx + sep.Length, e - idx - sep.Length));
            e = idx;
            c++;
        }
        r.Reverse();
        return r;
    }

    internal static string PyTitle(string s)
    {
        var sb = new StringBuilder(s.Length);
        bool prevCased = false;
        foreach (char c in s)
        {
            if (char.IsLetter(c))
            {
                sb.Append(prevCased ? char.ToLowerInvariant(c) : char.ToUpperInvariant(c));
                prevCased = true;
            }
            else
            {
                sb.Append(c);
                prevCased = false;
            }
        }
        return sb.ToString();
    }

    internal static string PyCapitalize(string s) =>
        s.Length == 0 ? s : char.ToUpperInvariant(s[0]) + s.Substring(1).ToLowerInvariant();

    internal static string PyCenter(string s, int width, char fill = ' ')
    {
        int marg = width - s.Length;
        if (marg <= 0)
        {
            return s;
        }

        int left = marg / 2 + (marg & width & 1);
        return new string(fill, left) + s + new string(fill, marg - left);
    }

    private static int NormIndex(object? v, int len, int def)
    {
        if (v is null or Undefined)
        {
            return def;
        }

        long i = Ops.ToLong(v);
        if (i < 0)
        {
            i = Math.Max(0, i + len);
        }

        return (int)Math.Min(i, len);
    }

    private static object? CallString(string s, string name, CallArgs a)
    {
        switch (name)
        {
            case "strip": return PyStrip(s, a.Get(0, "chars") as string, true, true);
            case "lstrip": return PyStrip(s, a.Get(0, "chars") as string, true, false);
            case "rstrip": return PyStrip(s, a.Get(0, "chars") as string, false, true);
            case "split": return PySplit(s, a.Get(0, "sep") as string, a.Get(1, "maxsplit", -1L) is long m ? m : -1);
            case "rsplit": return PyRSplit(s, a.Get(0, "sep") as string, a.Get(1, "maxsplit", -1L) is long m2 ? m2 : -1);
            case "splitlines":
            {
                var r = new List<object?>();
                int st = 0;
                for (int i = 0; i < s.Length; i++)
                {
                    if (s[i] == '\n' || s[i] == '\r')
                    {
                        r.Add(s.Substring(st, i - st));
                        if (s[i] == '\r' && i + 1 < s.Length && s[i + 1] == '\n')
                            {
                                i++;
                            }

                            st = i + 1;
                    }
                }
                if (st < s.Length)
                    {
                        r.Add(s.Substring(st));
                    }

                    return r;
            }
            case "startswith":
            case "endswith":
            {
                var p = a.Get(0, "prefix");
                int start = NormIndex(a.Get(1, "start"), s.Length, 0);
                int end = NormIndex(a.Get(2, "end"), s.Length, s.Length);
                string sub = start <= end ? s.Substring(start, end - start) : "";
                bool Test(object? x) => x is string t
                    ? (name == "startswith" ? sub.StartsWith(t, StringComparison.Ordinal) : sub.EndsWith(t, StringComparison.Ordinal))
                    : throw new TemplateRuntimeException($"{name} first arg must be str or a tuple of str, not {Ops.TypeName(x)}");
                if (p is List<object?> alts)
                    {
                        return alts.Any(Test);
                    }

                    return Test(p);
            }
            case "upper": return s.ToUpperInvariant();
            case "lower": return s.ToLowerInvariant();
            case "swapcase": return new string(s.Select(c => char.IsUpper(c) ? char.ToLowerInvariant(c) : char.ToUpperInvariant(c)).ToArray());
            case "capitalize": return PyCapitalize(s);
            case "title": return PyTitle(s);
            case "replace":
            {
                string o = Arg(a, 0, "old", name), n = Arg(a, 1, "new", name);
                long cnt = a.Get(2, "count", -1L) is long c ? c : -1;
                return Replace(s, o, n, cnt);
            }
            case "find":
            case "rfind":
            case "index":
            case "rindex":
            {
                string sub = Arg(a, 0, "sub", name);
                int start = NormIndex(a.Get(1, "start"), s.Length, 0);
                int end = NormIndex(a.Get(2, "end"), s.Length, s.Length);
                int idx = -1;
                if (start <= end)
                {
                    string window = s.Substring(start, end - start);
                    idx = name is "find" or "index" ? window.IndexOf(sub, StringComparison.Ordinal) : window.LastIndexOf(sub, StringComparison.Ordinal);
                    if (idx >= 0)
                        {
                            idx += start;
                        }
                    }
                if (idx < 0 && name is "index" or "rindex")
                    {
                        throw new TemplateRuntimeException("substring not found");
                    }

                    return (long)idx;
            }
            case "count":
            {
                string sub = Arg(a, 0, "sub", name);
                if (sub.Length == 0)
                    {
                        return (long)s.Length + 1;
                    }

                    long c = 0;
                int i = 0;
                while ((i = s.IndexOf(sub, i, StringComparison.Ordinal)) >= 0)
                {
                    c++;
                    i += sub.Length;
                }
                return c;
            }
            case "join":
            {
                var items = Ops.Iterate(a.Get(0, "iterable"));
                var parts = new List<string>(items.Count);
                foreach (var x in items)
                    {
                        parts.Add(x as string ?? throw new TemplateRuntimeException($"sequence item: expected str instance, {Ops.TypeName(x)} found"));
                    }

                    return string.Join(s, parts);
            }
            case "format": return PyFormat.StrFormat(s, a);
            case "isdigit":
            case "isnumeric":
            case "isdecimal": return s.Length > 0 && s.All(char.IsDigit);
            case "isalpha": return s.Length > 0 && s.All(char.IsLetter);
            case "isalnum": return s.Length > 0 && s.All(char.IsLetterOrDigit);
            case "isspace": return s.Length > 0 && s.All(char.IsWhiteSpace);
            case "islower": return s.Any(char.IsLetter) && !s.Any(char.IsUpper);
            case "isupper": return s.Any(char.IsLetter) && !s.Any(char.IsLower);
            case "zfill":
            {
                int w = (int)Ops.ToLong(a.Get(0, "width"));
                if (s.Length >= w)
                    {
                        return s;
                    }

                    bool sign = s.Length > 0 && (s[0] == '-' || s[0] == '+');
                return (sign ? s.Substring(0, 1) : "") + new string('0', w - s.Length) + (sign ? s.Substring(1) : s);
            }
            case "ljust":
            case "rjust":
            {
                int w = (int)Ops.ToLong(a.Get(0, "width"));
                string f = a.Get(1, "fillchar", " ") as string ?? " ";
                char fc = f.Length > 0 ? f[0] : ' ';
                return name == "ljust" ? s.PadRight(w, fc) : s.PadLeft(w, fc);
            }
            case "center":
            {
                string f = a.Get(1, "fillchar", " ") as string ?? " ";
                return PyCenter(s, (int)Ops.ToLong(a.Get(0, "width")), f.Length > 0 ? f[0] : ' ');
            }
            case "removeprefix":
            {
                string p = Arg(a, 0, "prefix", name);
                return p.Length > 0 && s.StartsWith(p, StringComparison.Ordinal) ? s.Substring(p.Length) : s;
            }
            case "removesuffix":
            {
                string p = Arg(a, 0, "suffix", name);
                return p.Length > 0 && s.EndsWith(p, StringComparison.Ordinal) ? s.Substring(0, s.Length - p.Length) : s;
            }
            case "partition":
            {
                string sep = Arg(a, 0, "sep", name);
                int i = s.IndexOf(sep, StringComparison.Ordinal);
                return i < 0
                    ? new JinjaTuple { s, "", "" }
                    : new JinjaTuple { s.Substring(0, i), sep, s.Substring(i + sep.Length) };
            }
            case "rpartition":
            {
                string sep = Arg(a, 0, "sep", name);
                int i = s.LastIndexOf(sep, StringComparison.Ordinal);
                return i < 0
                    ? new JinjaTuple { "", "", s }
                    : new JinjaTuple { s.Substring(0, i), sep, s.Substring(i + sep.Length) };
            }
            default: throw NoMethod(s, name);
        }
    }

    internal static string Replace(string s, string o, string n, long count)
    {
        if (count < 0)
        {
            return o.Length == 0 ? InsertBetween(s, n, long.MaxValue) : s.Replace(o, n, StringComparison.Ordinal);
        }

        if (o.Length == 0)
        {
            return InsertBetween(s, n, count);
        }

        var sb = new StringBuilder();
        int pos = 0;
        long done = 0;
        while (done < count)
        {
            int idx = s.IndexOf(o, pos, StringComparison.Ordinal);
            if (idx < 0)
            {
                break;
            }

            sb.Append(s, pos, idx - pos).Append(n);
            pos = idx + o.Length;
            done++;
        }
        sb.Append(s, pos, s.Length - pos);
        return sb.ToString();
    }

    private static string InsertBetween(string s, string n, long count)
    {
        var sb = new StringBuilder();
        long done = 0;
        for (int i = 0; i <= s.Length; i++)
        {
            if (done < count)
            {
                sb.Append(n);
                done++;
            }
            if (i < s.Length)
            {
                sb.Append(s[i]);
            }
        }
        return sb.ToString();
    }

    // ------------------------------------------------------------------ lists

    private static object? CallList(List<object?> l, string name, CallArgs a)
    {
        switch (name)
        {
            case "append":
                l.Add(a.Get(0, "object"));
                return null;
            case "extend":
                l.AddRange(Ops.Iterate(a.Get(0, "iterable")).ToArray());
                return null;
            case "insert":
            {
                long i = Ops.ToLong(a.Get(0, "index"));
                if (i < 0)
                    {
                        i = Math.Max(0, i + l.Count);
                    }

                    l.Insert((int)Math.Min(i, l.Count), a.Get(1, "object"));
                return null;
            }
            case "pop":
            {
                if (l.Count == 0)
                    {
                        throw new TemplateRuntimeException("pop from empty list");
                    }

                    long i = a.Has(0, "index") ? Ops.ToLong(a.Get(0, "index")) : -1;
                if (i < 0)
                    {
                        i += l.Count;
                    }

                    if (i < 0 || i >= l.Count)
                    {
                        throw new TemplateRuntimeException("pop index out of range");
                    }

                    var v = l[(int)i];
                l.RemoveAt((int)i);
                return v;
            }
            case "index":
            {
                var x = a.Get(0, "value");
                for (int i = 0; i < l.Count; i++)
                    {
                        if (Ops.Eq(l[i], x))
                        {
                            return (long)i;
                        }
                    }

                    throw new TemplateRuntimeException("value is not in list");
            }
            case "count":
            {
                var x = a.Get(0, "value");
                return (long)l.Count(e => Ops.Eq(e, x));
            }
            case "copy": return new List<object?>(l);
            case "reverse":
                l.Reverse();
                return null;
            case "clear":
                l.Clear();
                return null;
            case "remove":
            {
                var x = a.Get(0, "value");
                for (int i = 0; i < l.Count; i++)
                {
                    if (Ops.Eq(l[i], x))
                    {
                        l.RemoveAt(i);
                        return null;
                    }
                }
                throw new TemplateRuntimeException("list.remove(x): x not in list");
            }
            default: throw NoMethod(l, name);
        }
    }

    // ------------------------------------------------------------------ dicts

    private static object? CallDict(JinjaDict d, string name, CallArgs a)
    {
        switch (name)
        {
            case "items":
                return d.Select(kv => (object?)new JinjaTuple { kv.Key, kv.Value }).ToList();
            case "keys": return new List<object?>(d.Keys);
            case "values": return d.Select(kv => kv.Value).ToList();
            case "get":
            {
                var k = a.Get(0, "key");
                if (k is string or long or bool && d.TryGetValue(k, out var v))
                    {
                        return v;
                    }

                    return a.Get(1, "default");
            }
            case "pop":
            {
                var k = a.Get(0, "key");
                if (k is string or long or bool && d.TryGetValue(k, out var v))
                {
                    d.Remove(k);
                    return v;
                }
                if (a.Has(1, "default"))
                    {
                        return a.Get(1, "default");
                    }

                    throw new TemplateRuntimeException($"KeyError: {Ops.Repr(k)}");
            }
            case "update":
            {
                if (a.Positional.Count > 0 && a.Positional[0] is JinjaDict o)
                    {
                        foreach (var kv in o)
                        {
                            d[kv.Key] = kv.Value;
                        }
                    }

                    if (a.Named != null)
                    {
                        foreach (var kv in a.Named)
                        {
                            d[kv.Key] = kv.Value;
                        }
                    }

                    return null;
            }
            case "setdefault":
            {
                var k = a.Get(0, "key")!;
                if (d.TryGetValue(k, out var v))
                    {
                        return v;
                    }

                    var def = a.Get(1, "default");
                d[k] = def;
                return def;
            }
            case "copy": return d.Clone();
            case "clear":
                d.Clear();
                return null;
            default: throw NoMethod(d, name);
        }
    }
}
