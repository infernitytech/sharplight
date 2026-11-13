using System.Globalization;

namespace Sharplight.GgufDump;

/// <summary>Formats GGUF metadata values for a one-line display: long strings and arrays are abbreviated.</summary>
internal static class GgufValueFormatter
{
    private const int MaxStringPreviewLength = 100;
    private const int MaxArrayPreviewItems = 8;

    public static string Format(object value)
    {
        return value switch
        {
            string text => FormatString(text),
            Array array => FormatArray(array),
            _ => Convert.ToString(value, CultureInfo.InvariantCulture) ?? "",
        };
    }

    private static string FormatString(string text)
    {
        if (text.Length <= MaxStringPreviewLength)
        {
            return $"\"{Escape(text)}\"";
        }

        return $"\"{Escape(text[..MaxStringPreviewLength])}\"... ({text.Length} chars)";
    }

    private static string FormatArray(Array array)
    {
        int shown = Math.Min(array.Length, MaxArrayPreviewItems);
        var items = new List<string>(shown);
        for (int i = 0; i < shown; i++)
        {
            items.Add(Format(array.GetValue(i)!));
        }

        var ellipsis = array.Length > shown ? ", ..." : "";
        return $"{array.GetType().GetElementType()!.Name}[{array.Length}] [{string.Join(", ", items)}{ellipsis}]";
    }

    private static string Escape(string text)
    {
        return text.Replace("\n", "\\n").Replace("\r", "\\r").Replace("\t", "\\t");
    }
}
