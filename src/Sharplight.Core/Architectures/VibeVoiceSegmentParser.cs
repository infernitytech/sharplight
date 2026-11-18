using System.Globalization;
using System.Text.Json;

namespace Sharplight.Architectures;

/// <summary>
/// Reads the segments of VibeVoice-ASR's output as it grows: the text is <c>assistant</c> and a JSON array of segment objects (possibly cut off by the token budget); every call to
/// <see cref="Feed"/> with the text so far returns the objects that became complete since the last call.
/// </summary>
internal sealed class VibeVoiceSegmentParser
{
    private int _position = -1;

    /// <summary>The complete segments that are new in <paramref name="text"/>, the whole text decoded so far.</summary>
    /// <param name="text">The decoded output so far.</param>
    public IEnumerable<VibeVoiceSegment> Feed(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        if (_position < 0)
        {
            int open = text.IndexOf('[', StringComparison.Ordinal);
            if (open < 0)
            {
                yield break;
            }

            _position = open + 1;
        }

        while (true)
        {
            int start = _position;
            while (start < text.Length && (char.IsWhiteSpace(text[start]) || text[start] == ','))
            {
                start++;
            }

            if (start >= text.Length || text[start] != '{')
            {
                yield break;
            }

            int end = ObjectEnd(text, start);
            if (end < 0)
            {
                yield break;
            }

            _position = end + 1;
            if (Parse(text.AsSpan(start, end - start + 1)) is { } segment)
            {
                yield return segment;
            }
        }
    }

    // The index of the brace that closes the object starting at <paramref name="start"/>, honoring strings, or -1 when it is not complete yet.
    private static int ObjectEnd(string text, int start)
    {
        int depth = 0;
        bool inString = false;
        for (int i = start; i < text.Length; i++)
        {
            char c = text[i];
            if (inString)
            {
                if (c == '\\')
                {
                    i++;
                }
                else if (c == '"')
                {
                    inString = false;
                }

                continue;
            }

            if (c == '"')
            {
                inString = true;
            }
            else if (c == '{')
            {
                depth++;
            }
            else if (c == '}' && --depth == 0)
            {
                return i;
            }
        }

        return -1;
    }

    private static VibeVoiceSegment? Parse(ReadOnlySpan<char> json)
    {
        try
        {
            using var document = JsonDocument.Parse(json.ToString());
            var root = document.RootElement;
            string content = root.TryGetProperty("Content", out var c) && c.ValueKind == JsonValueKind.String ? c.GetString()! : "";
            return new VibeVoiceSegment(Number(root, "Start"), Number(root, "End"), (int)Number(root, "Speaker"), content);
        }
        catch (JsonException)
        {
            return null;
        }
    }

    private static double Number(JsonElement element, string name)
    {
        if (!element.TryGetProperty(name, out var value))
        {
            return 0;
        }

        return value.ValueKind == JsonValueKind.Number ? value.GetDouble()
            : value.ValueKind == JsonValueKind.String && double.TryParse(value.GetString(), NumberStyles.Float, CultureInfo.InvariantCulture, out double parsed) ? parsed : 0;
    }
}
