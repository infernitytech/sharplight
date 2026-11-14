namespace Sharplight.Studio.Services.Inspection;

/// <summary>Splits a node's formatted attributes (<c>{key=value, key=[1, 2], ...}</c>) into key/value pairs.</summary>
internal static class AttributeText
{
    /// <summary>
    /// The pairs in order. Commas inside brackets, braces, parentheses and angle brackets do not split; a part without <c>=</c> is
    /// returned with an empty key.
    /// </summary>
    public static IReadOnlyList<KeyValuePair<string, string>> Split(string attributes)
    {
        var text = attributes.Trim();
        if (text.Length >= 2 && text[0] == '{' && text[^1] == '}')
        {
            text = text[1..^1];
        }

        var result = new List<KeyValuePair<string, string>>();
        int depth = 0;
        int start = 0;
        for (int i = 0; i <= text.Length; i++)
        {
            char c = i < text.Length ? text[i] : ',';
            if (c is '[' or '{' or '(' or '<')
            {
                depth++;
            }
            else if (c is ']' or '}' or ')' or '>')
            {
                depth = Math.Max(0, depth - 1);
            }
            else if (c == ',' && depth == 0)
            {
                var part = text[start..i].Trim();
                start = i + 1;
                if (part.Length == 0)
                {
                    continue;
                }

                int equals = part.IndexOf('=', StringComparison.Ordinal);
                result.Add(equals > 0 ? new(part[..equals].Trim(), part[(equals + 1)..].Trim()) : new("", part));
            }
        }

        return result;
    }
}
