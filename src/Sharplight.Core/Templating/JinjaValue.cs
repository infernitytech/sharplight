namespace Sharplight.Templating;

/// <summary>Converts host objects into the template value model.</summary>
public static class JinjaValue
{
    /// <summary>
    /// Normalizes a host value: integers become long, floats double, char string, dictionaries with string keys
    /// become <see cref="JinjaDict"/>, other enumerables become lists. Existing template values pass through.
    /// </summary>
    public static object? From(object? v)
    {
        switch (v)
        {
            case null:
            case string:
            case Markup:
            case bool:
            case long:
            case double:
            case Undefined:
            case JinjaDict:
            case JinjaNamespace:
            case JinjaCallable:
                return v;
            case List<object?> l:
                return l;
            case int i: return (long)i;
            case short s: return (long)s;
            case sbyte sb: return (long)sb;
            case byte b: return (long)b;
            case ushort us: return (long)us;
            case uint ui: return (long)ui;
            case ulong ul: return ul <= long.MaxValue ? (long)ul : (double)ul;
            case float f: return (double)f;
            case decimal m: return (double)m;
            case char c: return c.ToString();
            case Enum e: return e.ToString();
            case IReadOnlyDictionary<string, object?> rd:
            {
                var d = new JinjaDict();
                foreach (var kv in rd)
                    {
                        d[kv.Key] = From(kv.Value);
                    }

                    return d;
            }
            case IDictionary<string, object?> dd:
            {
                var d = new JinjaDict();
                foreach (var kv in dd)
                    {
                        d[kv.Key] = From(kv.Value);
                    }

                    return d;
            }
            case System.Collections.IDictionary nd:
            {
                var d = new JinjaDict();
                foreach (System.Collections.DictionaryEntry kv in nd)
                    {
                        d[From(kv.Key)!] = From(kv.Value);
                    }

                    return d;
            }
            case System.Collections.IEnumerable en:
            {
                var l = new List<object?>();
                foreach (var x in en)
                    {
                        l.Add(From(x));
                    }

                    return l;
            }
            default:
                return v.ToString();
        }
    }
}
