using System.Globalization;
using System.Text;

namespace Sharplight.Templating;

/// <summary>Builtin filters, tests and global functions.</summary>
internal static class Builtins
{
    // ------------------------------------------------------------------ helpers

    private static string S(object? v) => Ops.Str(v);

    private static object? AttrPath(object? o, string path, bool strict)
    {
        foreach (var part in path.Split('.'))
        {
            object? next = Ops.GetItem(o, long.TryParse(part, out var idx) ? idx : part, strict);
            if (next is Undefined)
            {
                next = Ops.GetAttr(o, part, strict);
            }

            o = next;
        }
        return o;
    }

    private static object? AttrOf(JinjaEnvironment env, object? item, object? attr)
    {
        if (attr is string s)
        {
            return AttrPath(item, s, env.StrictUndefined);
        }

        if (attr is long l)
        {
            return Ops.GetItem(item, l, env.StrictUndefined);
        }

        return item;
    }

    private static object? SortKey(object? v, bool caseSensitive) =>
        !caseSensitive && v is string s ? s.ToLowerInvariant() : v;

    private sealed class ValueComparer : IComparer<object?>
    {
        public static readonly ValueComparer Instance = new();
        public int Compare(object? x, object? y) => Ops.Compare(x, y);
    }

    private static List<object?> Seq(object? v) => Ops.Iterate(v);

    private static bool Boolean(CallArgs a, int i, string name, bool def) => Ops.Truthy(a.Get(i, name, def));

    private static object? CallTest(JinjaEnvironment env, string name, object? value, List<object?> extra)
    {
        if (!env.Tests.TryGetValue(name, out var t))
        {
            throw new TemplateRuntimeException($"No test named '{name}'");
        }

        return t(env, value, new CallArgs(extra, null));
    }

    private static string Indent(object? v)
    {
        if (v is null or Undefined)
        {
            return "";
        }

        if (v is string s)
        {
            return s;
        }

        return new string(' ', (int)Ops.ToLong(v));
    }

    // ------------------------------------------------------------------ registration

    public static Dictionary<string, FilterFunction> CreateFilters()
    {
        var f = new Dictionary<string, FilterFunction>();

        f["length"] = f["count"] = (env, v, a) => (long)Ops.Len(v);
        f["upper"] = (env, v, a) => S(v).ToUpperInvariant();
        f["lower"] = (env, v, a) => S(v).ToLowerInvariant();
        f["capitalize"] = (env, v, a) => Methods.PyCapitalize(S(v));
        f["title"] = (env, v, a) => JinjaTitle(S(v));
        f["trim"] = (env, v, a) => Methods.PyStrip(S(v), a.Get(0, "chars") as string, true, true);
        f["string"] = (env, v, a) => S(v);
        f["safe"] = (env, v, a) => v as Markup ?? new Markup(S(v));
        f["escape"] = f["e"] = (env, v, a) => v as Markup ?? new Markup(Markup.Escape(S(v)));
        f["forceescape"] = (env, v, a) => new Markup(Markup.Escape(S(v)));
        f["striptags"] = (env, v, a) => System.Text.RegularExpressions.Regex.Replace(S(v), "<[^>]*>", "").Trim();
        f["pprint"] = (env, v, a) => Ops.Repr(v);

        f["default"] = f["d"] = (env, v, a) =>
        {
            var def = a.Get(0, "default_value", "");
            bool boolean = Boolean(a, 1, "boolean", false);
            if (v is Undefined)
            {
                return def;
            }

            if (boolean && !Ops.Truthy(v))
            {
                return def;
            }

            return v;
        };

        f["first"] = (env, v, a) =>
        {
            if (v is string s)
            {
                return s.Length > 0 ? s.Substring(0, char.IsHighSurrogate(s[0]) && s.Length > 1 ? 2 : 1) : new Undefined("first", env.StrictUndefined);
            }

            var l = Seq(v);
            return l.Count > 0 ? l[0] : new Undefined("first", env.StrictUndefined);
        };
        f["last"] = (env, v, a) =>
        {
            if (v is string s)
            {
                return s.Length > 0 ? s.Substring(s.Length - (char.IsLowSurrogate(s[^1]) && s.Length > 1 ? 2 : 1)) : new Undefined("last", env.StrictUndefined);
            }

            var l = Seq(v);
            return l.Count > 0 ? l[^1] : new Undefined("last", env.StrictUndefined);
        };
        f["join"] = (env, v, a) =>
        {
            string sep = a.Get(0, "d", "") is string ss ? ss : "";
            var attr = a.Get(1, "attribute");
            var sb = new StringBuilder();
            bool first = true;
            foreach (var x in Seq(v))
            {
                if (!first)
                {
                    sb.Append(sep);
                }

                first = false;
                sb.Append(S(attr == null ? x : AttrOf(env, x, attr)));
            }
            return sb.ToString();
        };
        f["replace"] = (env, v, a) =>
        {
            string o = S(a.Get(0, "old"));
            string n = S(a.Get(1, "new"));
            long cnt = a.Get(2, "count") is long c ? c : -1;
            return Methods.Replace(S(v), o, n, cnt);
        };
        f["tojson"] = (env, v, a) =>
        {
            // HF semantics: json.dumps(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False)
            bool ascii = Boolean(a, 0, "ensure_ascii", false);
            var ind = a.Get(1, "indent");
            string? indent = ind is null or Undefined ? null : Indent(ind);
            string? item = null;
            string? key = null;
            if (a.Get(2, "separators") is List<object?> sep && sep.Count == 2)
            {
                item = S(sep[0]);
                key = S(sep[1]);
            }
            bool sort = Boolean(a, 3, "sort_keys", false);
            return JinjaJson.Dumps(v, ascii, indent, item, key, sort);
        };
        f["from_json"] = f["fromjson"] = (env, v, a) => JinjaJson.Parse(S(v));

        f["int"] = (env, v, a) =>
        {
            var def = a.Get(0, "default", 0L);
            switch (v)
            {
                case long: return v;
                case bool b: return b ? 1L : 0L;
                case double d: return double.IsFinite(d) ? (long)d : def;
                case string s:
                {
                    s = s.Trim();
                    int bas = a.Get(1, "base", 10L) is long bl ? (int)bl : 10;
                    if (bas == 10)
                    {
                        if (long.TryParse(s, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out var l))
                            {
                                return l;
                            }

                            if (double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out var d2) && double.IsFinite(d2))
                            {
                                return (long)d2;
                            }
                        }
                    else
                    {
                        try { return Convert.ToInt64(s, bas); } catch (Exception) { }
                    }
                    return def;
                }
                default: return def;
            }
        };
        f["float"] = (env, v, a) =>
        {
            var def = a.Get(0, "default", 0.0);
            switch (v)
            {
                case double: return v;
                case long or bool: return Ops.ToDouble(v);
                case string s when double.TryParse(s.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out var d): return d;
                default: return def;
            }
        };
        f["abs"] = (env, v, a) => v is double d ? Math.Abs(d) : Ops.IsInt(v) ? (object)Math.Abs(Ops.ToLong(v)) : throw new TemplateRuntimeException($"bad operand type for abs(): '{Ops.TypeName(v)}'");
        f["round"] = (env, v, a) =>
        {
            int prec = (int)Ops.ToLong(a.Get(0, "precision", 0L));
            string method = S(a.Get(1, "method", "common"));
            if (v is long or bool)
            {
                long n = Ops.ToLong(v);
                if (prec >= 0)
                {
                    return n;
                }

                double p = Math.Pow(10, -prec);
                double q = method switch { "ceil" => Math.Ceiling(n / p), "floor" => Math.Floor(n / p), _ => Math.Round(n / p, MidpointRounding.ToEven) };
                return (long)(q * p);
            }
            double d = Ops.ToDouble(v);
            return method switch
            {
                "ceil" => Math.Ceiling(d * Math.Pow(10, prec)) / Math.Pow(10, prec),
                "floor" => Math.Floor(d * Math.Pow(10, prec)) / Math.Pow(10, prec),
                _ => prec >= 0 && prec <= 15 ? Math.Round(d, prec, MidpointRounding.ToEven) : Math.Round(d / Math.Pow(10, -prec), MidpointRounding.ToEven) * Math.Pow(10, -prec),
            };
        };

        f["list"] = (env, v, a) => new List<object?>(Seq(v));
        f["items"] = (env, v, a) => v switch
        {
            JinjaDict d => d.Select(kv => (object?)new JinjaTuple { kv.Key, kv.Value }).ToList(),
            Undefined u when !u.Strict => new List<object?>(),
            Undefined u => throw u.Fail(),
            _ => throw new TemplateRuntimeException($"'items' filter requires a mapping, got {Ops.TypeName(v)}"),
        };
        f["keys"] = (env, v, a) => v is JinjaDict d ? new List<object?>(d.Keys) : throw new TemplateRuntimeException("'keys' filter requires a mapping");
        f["values"] = (env, v, a) => v is JinjaDict d ? d.Select(kv => kv.Value).ToList() : throw new TemplateRuntimeException("'values' filter requires a mapping");
        f["attr"] = (env, v, a) =>
        {
            string name = S(a.Get(0, "name"));
            if (v is JinjaDict d)
            {
                return Methods.HasDictMethod(name) ? new BoundMethod(d, name) : new Undefined(name, env.StrictUndefined);
            }

            return Ops.GetAttr(v, name, env.StrictUndefined);
        };

        f["reverse"] = (env, v, a) =>
        {
            if (v is string s)
            {
                var runes = s.EnumerateRunes().Reverse().Select(r => r.ToString());
                return string.Concat(runes);
            }
            var l = new List<object?>(Seq(v));
            l.Reverse();
            return l;
        };
        f["sum"] = (env, v, a) =>
        {
            var attr = a.Get(1, "attribute");
            object? acc = a.Get(2, "start", 0L);
            foreach (var x in Seq(v))
            {
                acc = Ops.Add(acc, attr == null ? x : AttrOf(env, x, attr));
            }

            return acc;
        };
        f["min"] = (env, v, a) => MinMax(env, v, a, -1);
        f["max"] = (env, v, a) => MinMax(env, v, a, 1);
        f["unique"] = (env, v, a) =>
        {
            bool cs = Boolean(a, 0, "case_sensitive", false);
            var attr = a.Get(1, "attribute");
            var seen = new List<object?>();
            var res = new List<object?>();
            foreach (var x in Seq(v))
            {
                var k = SortKey(attr == null ? x : AttrOf(env, x, attr), cs);
                if (seen.Any(s => Ops.Eq(s, k)))
                {
                    continue;
                }

                seen.Add(k);
                res.Add(x);
            }
            return res;
        };
        f["sort"] = (env, v, a) =>
        {
            bool rev = Boolean(a, 0, "reverse", false);
            bool cs = Boolean(a, 1, "case_sensitive", false);
            var attr = a.Get(2, "attribute");
            var src = Seq(v);
            object? Key(object? x) => SortKey(attr == null ? x : AttrOf(env, x, attr), cs);
            var sorted = rev ? src.OrderByDescending(Key, ValueComparer.Instance) : src.OrderBy(Key, ValueComparer.Instance);
            return sorted.ToList();
        };
        f["dictsort"] = (env, v, a) =>
        {
            if (v is not JinjaDict d)
            {
                throw new TemplateRuntimeException("dictsort requires a mapping");
            }

            bool cs = Boolean(a, 0, "case_sensitive", false);
            bool byValue = S(a.Get(1, "by", "key")) == "value";
            bool rev = Boolean(a, 2, "reverse", false);
            var pairs = d.Select(kv => (object?)new JinjaTuple { kv.Key, kv.Value });
            object? Key(object? p)
            {
                var l = (List<object?>)p!;
                return SortKey(l[byValue ? 1 : 0], cs);
            }
            return (rev ? pairs.OrderByDescending(Key, ValueComparer.Instance) : pairs.OrderBy(Key, ValueComparer.Instance)).ToList();
        };
        f["batch"] = (env, v, a) =>
        {
            int n = (int)Ops.ToLong(a.Get(0, "linecount"));
            var fill = a.Get(1, "fill_with");
            bool hasFill = a.Has(1, "fill_with") && fill != null;
            var res = new List<object?>();
            List<object?>? cur = null;
            foreach (var x in Seq(v))
            {
                if (cur != null && cur.Count == n) { res.Add(cur); cur = null; }
                (cur ??= new List<object?>()).Add(x);
            }
            if (cur != null)
            {
                if (hasFill)
                {
                    while (cur.Count < n)
                    {
                        cur.Add(fill);
                    }
                }

                res.Add(cur);
            }
            return res;
        };
        f["slice"] = (env, v, a) =>
        {
            int n = (int)Ops.ToLong(a.Get(0, "slices"));
            var fill = a.Get(1, "fill_with");
            var seq = Seq(v);
            int per = seq.Count / n;
            int extra = seq.Count % n;
            int offset = 0;
            var res = new List<object?>();
            for (int i = 0; i < n; i++)
            {
                int start = offset + i * per;
                if (i < extra)
                {
                    offset++;
                }

                int end = offset + (i + 1) * per;
                var tmp = seq.GetRange(start, end - start);
                if (a.Has(1, "fill_with") && i >= extra)
                {
                    tmp.Add(fill);
                }

                res.Add(tmp);
            }
            return res;
        };
        f["groupby"] = (env, v, a) =>
        {
            var attr = a.Get(0, "attribute");
            var groups = new List<(object? Key, List<object?> Items)>();
            var sorted = Seq(v).OrderBy(x => AttrOf(env, x, attr), ValueComparer.Instance);
            foreach (var x in sorted)
            {
                var k = AttrOf(env, x, attr);
                if (groups.Count > 0 && Ops.Eq(groups[^1].Key, k))
                {
                    groups[^1].Items.Add(x);
                }
                else
                {
                    groups.Add((k, new List<object?> { x }));
                }
            }
            return groups.Select(g => (object?)new JinjaTuple { g.Key, g.Items }).ToList();
        };

        f["map"] = (env, v, a) =>
        {
            var res = new List<object?>();
            if (!Ops.Truthy(v))
            {
                return res;
            }

            if (a.Named != null && a.Named.ContainsKey("attribute"))
            {
                var attr = a.Named["attribute"];
                bool hasDef = a.Named.TryGetValue("default", out var def);
                foreach (var x in Seq(v))
                {
                    var r = AttrOf(env, x, attr);
                    res.Add(r is Undefined && hasDef ? def : r);
                }
                return res;
            }
            if (a.Positional.Count == 0 || a.Positional[0] is not string fname)
            {
                throw new TemplateRuntimeException("map requires a filter name or attribute=");
            }

            if (!env.Filters.TryGetValue(fname, out var fn))
            {
                throw new TemplateRuntimeException($"No filter named '{fname}'");
            }

            var rest = new CallArgs(a.Positional.Skip(1).ToList(), a.Named);
            foreach (var x in Seq(v))
            {
                res.Add(fn(env, x, rest));
            }

            return res;
        };
        f["select"] = (env, v, a) => SelectCore(env, v, a, 0, false, false);
        f["reject"] = (env, v, a) => SelectCore(env, v, a, 0, true, false);
        f["selectattr"] = (env, v, a) => SelectCore(env, v, a, 1, false, true);
        f["rejectattr"] = (env, v, a) => SelectCore(env, v, a, 1, true, true);

        f["indent"] = (env, v, a) =>
        {
            string s = S(v);
            string pad = Indent(a.Get(0, "width", 4L));
            bool first = Boolean(a, 1, "first", false);
            bool blank = Boolean(a, 2, "blank", false);
            var lines = s.Split('\n');
            var sb = new StringBuilder();
            for (int i = 0; i < lines.Length; i++)
            {
                if (i > 0)
                {
                    sb.Append('\n');
                }

                if (lines[i].Length == 0 && !blank)
                {
                    continue;
                }

                if (i == 0 && !first) { sb.Append(lines[i]); continue; }
                sb.Append(pad).Append(lines[i]);
            }
            return sb.ToString();
        };
        f["center"] = (env, v, a) => Methods.PyCenter(S(v), (int)Ops.ToLong(a.Get(0, "width", 80L)));
        f["wordcount"] = (env, v, a) =>
        {
            long n = 0;
            bool inWord = false;
            foreach (char c in S(v))
            {
                bool w = char.IsLetterOrDigit(c) || c == '_';
                if (w && !inWord)
                {
                    n++;
                }

                inWord = w;
            }
            return n;
        };
        f["truncate"] = (env, v, a) =>
        {
            string s = S(v);
            int len = (int)Ops.ToLong(a.Get(0, "length", 255L));
            bool kill = Boolean(a, 1, "killwords", false);
            string end = S(a.Get(2, "end", "..."));
            int leeway = (int)Ops.ToLong(a.Get(3, "leeway", 5L));
            if (s.Length <= len + leeway)
            {
                return s;
            }

            string cut = s.Substring(0, Math.Max(0, len - end.Length));
            if (kill)
            {
                return cut + end;
            }

            int sp = cut.LastIndexOf(' ');
            return (sp >= 0 ? cut.Substring(0, sp) : cut) + end;
        };
        f["urlencode"] = (env, v, a) =>
        {
            var sb = new StringBuilder();
            foreach (byte b in Encoding.UTF8.GetBytes(S(v)))
            {
                char c = (char)b;
                if (char.IsAsciiLetterOrDigit(c) || c is '_' or '.' or '-' or '~' or '/')
                {
                    sb.Append(c);
                }
                else
                {
                    sb.Append('%').Append(b.ToString("X2", CultureInfo.InvariantCulture));
                }
            }
            return sb.ToString();
        };
        f["format"] = (env, v, a) => PyFormat.Percent(S(v), new List<object?>(a.Positional));
        f["xmlattr"] = (env, v, a) =>
        {
            if (v is not JinjaDict d)
            {
                throw new TemplateRuntimeException("xmlattr requires a mapping");
            }

            var sb = new StringBuilder();
            foreach (var kv in d)
            {
                if (kv.Value is not (null or Undefined))
                {
                    sb.Append(' ').Append(S(kv.Key)).Append("=\"").Append(Markup.Escape(S(kv.Value))).Append('"');
                }
            }

            return sb.Length > 0 ? sb.ToString() : "";
        };
        return f;
    }

    private static string JinjaTitle(string s)
    {
        var sb = new StringBuilder(s.Length);
        bool start = true;
        foreach (char c in s)
        {
            if (start)
            {
                sb.Append(char.ToUpperInvariant(c));
            }
            else
            {
                sb.Append(char.ToLowerInvariant(c));
            }

            start = char.IsWhiteSpace(c) || c is '-' or '(' or '{' or '[' or '<';
        }
        return sb.ToString();
    }

    private static object? MinMax(JinjaEnvironment env, object? v, CallArgs a, int dir)
    {
        bool cs = Boolean(a, 0, "case_sensitive", false);
        var attr = a.Get(1, "attribute");
        var seq = Seq(v);
        if (seq.Count == 0)
        {
            return new Undefined(dir < 0 ? "min" : "max", env.StrictUndefined);
        }

        object? best = seq[0];
        object? bestKey = SortKey(attr == null ? best : AttrOf(env, best, attr), cs);
        for (int i = 1; i < seq.Count; i++)
        {
            var k = SortKey(attr == null ? seq[i] : AttrOf(env, seq[i], attr), cs);
            if (Ops.Compare(k, bestKey) * dir > 0)
            {
                best = seq[i];
                bestKey = k;
            }
        }
        return best;
    }

    private static object? SelectCore(JinjaEnvironment env, object? v, CallArgs a, int testIdx, bool reject, bool byAttr)
    {
        object? attr = byAttr ? a.Get(0, "attribute") : null;
        string? testName = a.Positional.Count > testIdx ? a.Positional[testIdx] as string : null;
        var extra = a.Positional.Count > testIdx + 1 ? a.Positional.GetRange(testIdx + 1, a.Positional.Count - testIdx - 1) : new List<object?>();
        var res = new List<object?>();
        if (!Ops.Truthy(v))
        {
            return res;
        }

        foreach (var x in Seq(v))
        {
            var target = byAttr ? AttrOf(env, x, attr) : x;
            bool ok = testName == null ? Ops.Truthy(target) : (bool)CallTest(env, testName, target, extra)!;
            if (ok != reject)
            {
                res.Add(x);
            }
        }
        return res;
    }

    // ------------------------------------------------------------------ tests

    public static Dictionary<string, TestFunction> CreateTests()
    {
        var t = new Dictionary<string, TestFunction>();
        t["defined"] = (env, v, a) => v is not Undefined;
        t["undefined"] = (env, v, a) => v is Undefined;
        t["none"] = (env, v, a) => v is null;
        t["string"] = (env, v, a) => v is string or Markup;
        t["mapping"] = (env, v, a) => v is JinjaDict;
        t["iterable"] = (env, v, a) => v is string or Markup or List<object?> or JinjaDict;
        t["sequence"] = (env, v, a) => v is string or Markup or List<object?> or JinjaDict;
        t["number"] = (env, v, a) => Ops.IsNum(v);
        t["integer"] = (env, v, a) => v is long;
        t["float"] = (env, v, a) => v is double;
        t["boolean"] = (env, v, a) => v is bool;
        t["true"] = (env, v, a) => v is true;
        t["false"] = (env, v, a) => v is false;
        t["callable"] = (env, v, a) => v is JinjaCallable;
        t["odd"] = (env, v, a) => Ops.IsInt(v) ? Ops.ToLong(v) % 2 != 0 : throw new TemplateRuntimeException("'odd' test requires an integer");
        t["even"] = (env, v, a) => Ops.IsInt(v) ? Ops.ToLong(v) % 2 == 0 : throw new TemplateRuntimeException("'even' test requires an integer");
        t["divisibleby"] = (env, v, a) =>
        {
            long n = Ops.ToLong(a.Get(0, "num"));
            if (n == 0)
            {
                throw new TemplateRuntimeException("division by zero");
            }

            return Ops.ToLong(v) % n == 0;
        };
        t["eq"] = t["equalto"] = t["=="] = (env, v, a) => Ops.Eq(v, a.Get(0, "other"));
        t["ne"] = t["!="] = (env, v, a) => !Ops.Eq(v, a.Get(0, "other"));
        t["lt"] = t["lessthan"] = t["<"] = (env, v, a) => Ops.Compare(v, a.Get(0, "other"), "<") < 0;
        t["le"] = t["<="] = (env, v, a) => Ops.Compare(v, a.Get(0, "other"), "<=") <= 0;
        t["gt"] = t["greaterthan"] = t[">"] = (env, v, a) => Ops.Compare(v, a.Get(0, "other"), ">") > 0;
        t["ge"] = t[">="] = (env, v, a) => Ops.Compare(v, a.Get(0, "other"), ">=") >= 0;
        t["in"] = (env, v, a) => Ops.Contains(a.Get(0, "seq"), v);
        t["sameas"] = (env, v, a) =>
        {
            var o = a.Get(0, "other");
            if (v is bool b1 && o is bool b2)
            {
                return b1 == b2;
            }

            if (v is null)
            {
                return o is null;
            }

            if (v is Undefined)
            {
                return o is Undefined;
            }

            if (v is long l1 && o is long l2)
            {
                return l1 == l2;
            }

            if (v is string s1 && o is string s2)
            {
                return ReferenceEquals(s1, s2) || s1 == s2;
            }

            return ReferenceEquals(v, o);
        };
        t["lower"] = (env, v, a) => v is string s && s.Any(char.IsLetter) && !s.Any(char.IsUpper);
        t["upper"] = (env, v, a) => v is string s && s.Any(char.IsLetter) && !s.Any(char.IsLower);
        t["escaped"] = (env, v, a) => false;
        return t;
    }

    // ------------------------------------------------------------------ globals

    public static void AddGlobals(JinjaEnvironment env)
    {
        var g = env.Globals;
        g["range"] = new NativeFunction("range", a =>
        {
            long start = 0;
            long stop;
            long step = 1;
            var p = a.Positional;
            if (p.Count == 1)
            {
                stop = Ops.ToLong(p[0]);
            }
            else if (p.Count >= 2)
            {
                start = Ops.ToLong(p[0]);
                stop = Ops.ToLong(p[1]);
                if (p.Count >= 3)
                {
                    step = Ops.ToLong(p[2]);
                }
            }
            else
            {
                throw new TemplateRuntimeException("range expected at least 1 argument, got 0");
            }

            if (step == 0)
            {
                throw new TemplateRuntimeException("range() arg 3 must not be zero");
            }

            var l = new List<object?>();
            if (step > 0)
            {
                for (long i = start; i < stop; i += step)
                {
                    l.Add(i);
                }
            }
            else
            {
                for (long i = start; i > stop; i += step)
                {
                    l.Add(i);
                }
            }

            return l;
        });
        g["namespace"] = new NativeFunction("namespace", a =>
        {
            var ns = new JinjaNamespace();
            if (a.Positional.Count > 0 && a.Positional[0] is JinjaDict d)
            {
                foreach (var kv in d)
                {
                    ns.Attributes[Ops.Str(kv.Key)] = kv.Value;
                }
            }

            if (a.Named != null)
            {
                foreach (var kv in a.Named)
                {
                    ns.Attributes[kv.Key] = kv.Value;
                }
            }

            return ns;
        });
        g["dict"] = new NativeFunction("dict", a =>
        {
            var d = new JinjaDict();
            if (a.Positional.Count > 0)
            {
                var src = a.Positional[0];
                if (src is JinjaDict sd)
                {
                    foreach (var kv in sd)
                    {
                        d[kv.Key] = kv.Value;
                    }
                }
                else
                {
                    foreach (var pair in Ops.Iterate(src))
                    {
                        var pl = Ops.Iterate(pair);
                        if (pl.Count != 2)
                        {
                            throw new TemplateRuntimeException("dict() sequence element has wrong length");
                        }

                        d[pl[0]!] = pl[1];
                    }
                }
            }
            if (a.Named != null)
            {
                foreach (var kv in a.Named)
                {
                    d[kv.Key] = kv.Value;
                }
            }

            return d;
        });
        g["raise_exception"] = new NativeFunction("raise_exception", a =>
            throw new TemplateRaisedException(a.Positional.Count > 0 ? Ops.Str(a.Positional[0]) : "raise_exception called"));
        g["strftime_now"] = new NativeFunction("strftime_now", a => PyFormat.Strftime(Ops.Str(a.Get(0, "format")), env.Clock()));
        g["len"] = new NativeFunction("len", a => (long)Ops.Len(a.Get(0, "obj")));
        g["lipsum"] = new NativeFunction("lipsum", a => "Lorem ipsum dolor sit amet, consectetur adipiscing elit.");
    }
}
