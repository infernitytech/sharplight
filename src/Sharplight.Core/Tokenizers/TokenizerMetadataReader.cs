using System.Collections;
using System.Globalization;

namespace Sharplight.Tokenizers;

/// <summary>Lenient readers for canonical metadata values: numeric widths and array element types vary between model formats.</summary>
internal static class TokenizerMetadataReader
{
    public static string? GetString(IReadOnlyDictionary<string, object> md, string key)
        => md.TryGetValue(key, out var v) ? v as string ?? v?.ToString() : null;

    /// <summary>Reads an id; negative or out-of-range values (llama.cpp's 0xFFFFFFFF "none") yield null.</summary>
    public static int? GetInt(IReadOnlyDictionary<string, object> md, string key)
    {
        if (!md.TryGetValue(key, out var v) || v is null)
        {
            return null;
        }

        long n = ToLong(v);
        return n is < 0 or > int.MaxValue ? null : (int)n;
    }

    public static bool? GetBool(IReadOnlyDictionary<string, object> md, string key)
    {
        if (!md.TryGetValue(key, out var v) || v is null)
        {
            return null;
        }

        return v is bool b ? b : ToLong(v) != 0;
    }

    public static string[]? GetStringArray(IReadOnlyDictionary<string, object> md, string key)
    {
        if (!md.TryGetValue(key, out var v) || v is null)
        {
            return null;
        }

        if (v is string[] strings)
        {
            return strings;
        }

        if (v is string || v is not IEnumerable items)
        {
            return null;
        }

        var list = new List<string>();
        foreach (object? item in items)
        {
            list.Add(item as string ?? item?.ToString() ?? "");
        }

        return list.ToArray();
    }

    public static float[]? GetFloatArray(IReadOnlyDictionary<string, object> md, string key)
    {
        if (!md.TryGetValue(key, out var v) || v is null)
        {
            return null;
        }

        if (v is float[] floats)
        {
            return floats;
        }

        if (v is string || v is not IEnumerable items)
        {
            return null;
        }

        var list = new List<float>();
        foreach (object? item in items)
        {
            list.Add(Convert.ToSingle(item, CultureInfo.InvariantCulture));
        }

        return list.ToArray();
    }

    public static int[]? GetIntArray(IReadOnlyDictionary<string, object> md, string key)
    {
        if (!md.TryGetValue(key, out var v) || v is null)
        {
            return null;
        }

        if (v is int[] ints)
        {
            return ints;
        }

        if (v is string || v is not IEnumerable items)
        {
            return null;
        }

        var list = new List<int>();
        foreach (object? item in items)
        {
            list.Add(checked((int)ToLong(item!)));
        }

        return list.ToArray();
    }

    private static long ToLong(object v) => v switch
    {
        long l => l,
        int i => i,
        uint u => u,
        ulong ul => ul > long.MaxValue ? -1 : (long)ul,
        short s => s,
        ushort us => us,
        byte b => b,
        sbyte sb => sb,
        bool bo => bo ? 1 : 0,
        float f => (long)f,
        double d => (long)d,
        IConvertible c => c.ToInt64(CultureInfo.InvariantCulture),
        _ => throw new InvalidDataException($"Cannot interpret {v.GetType()} as a number."),
    };
}
