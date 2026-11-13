using System.Globalization;

namespace Sharplight.OnnxRun;

/// <summary>Parses <c>--shape name=-1x3x224x224</c> values (a non-positive dimension marks the batch axis).</summary>
internal static class InputShapeParser
{
    private const char NameSeparator = '=';
    private const char DimensionSeparator = 'x';

    /// <summary>Parses one value; on failure returns false and describes the problem in <paramref name="error"/>.</summary>
    public static bool TryParse(string text, out string name, out long[] dimensions, out string? error)
    {
        name = "";
        dimensions = [];
        int separator = text.IndexOf(NameSeparator);
        if (separator <= 0)
        {
            error = $"--shape expects NAME=DIMxDIM..., got '{text}'.";
            return false;
        }

        var parsed = new List<long>();
        foreach (var part in text[(separator + 1)..].Split(DimensionSeparator))
        {
            if (!long.TryParse(part, NumberStyles.Integer, CultureInfo.InvariantCulture, out long dimension))
            {
                error = $"--shape '{text}': '{part}' is not an integer dimension.";
                return false;
            }

            parsed.Add(dimension);
        }

        name = text[..separator];
        dimensions = [.. parsed];
        error = null;
        return true;
    }
}
