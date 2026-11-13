using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace Sharplight.Llm.Serving.Chat;

/// <summary>Built-in formats and format detection from a chat template.</summary>
public static partial class ToolCallFormats
{
    /// <summary><c>&lt;tool_call&gt;{json}&lt;/tool_call&gt;</c> (Hermes, Qwen).</summary>
    public static IToolCallFormat Hermes { get; } = new HermesFormat();
    /// <summary><c>[TOOL_CALLS][{json}, ...]</c> (Mistral).</summary>
    public static IToolCallFormat Mistral { get; } = new MistralFormat();
    /// <summary><c>&lt;|python_tag|&gt;{json}</c> or a bare JSON object (Llama 3.x).</summary>
    public static IToolCallFormat Llama3 { get; } = new Llama3Format();
    /// <summary>All built-in formats.</summary>
    public static IReadOnlyList<IToolCallFormat> All { get; } = [Hermes, Mistral, Llama3];

    /// <summary>Picks the formats a chat template instructs the model to use; all built-in ones when it is unknown.</summary>
    public static IReadOnlyList<IToolCallFormat> ForTemplate(string? template)
    {
        if (template is null)
        {
            return All;
        }

        if (template.Contains("<tool_call>", StringComparison.Ordinal))
        {
            return [Hermes];
        }

        if (template.Contains("[TOOL_CALLS]", StringComparison.Ordinal))
        {
            return [Mistral];
        }

        if (template.Contains("<|python_tag|>", StringComparison.Ordinal) || template.Contains("ipython", StringComparison.Ordinal))
        {
            return [Llama3];
        }

        return All;
    }

    internal static string JsonString(string s) => "\"" + JsonEncodedText.Encode(s, System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping) + "\"";

    /// <summary>Parses one or several JSON call objects (<c>{"name":..,"arguments"|"parameters":{..}}</c>, an array of them, or objects separated by ';'/newlines).</summary>
    public static bool TryParseJsonCalls(string text, out IReadOnlyList<ParsedCall> calls)
    {
        var res = new List<ParsedCall>();
        calls = res;
        text = text.Trim();
        if (text.Length == 0)
        {
            return false;
        }

        foreach (var piece in SplitTopLevel(text))
        {
            JsonDocument doc;
            try { doc = JsonDocument.Parse(piece); } catch (JsonException) { return false; }
            using (doc)
            {
                var root = doc.RootElement;
                if (root.ValueKind == JsonValueKind.Array)
                {
                    foreach (var el in root.EnumerateArray())
                    {
                        if (!AddCall(el, res))
                        {
                            return false;
                        }
                    }
                }
                else if (!AddCall(root, res))
                {
                    return false;
                }
            }
        }
        return res.Count > 0;
    }

    private static bool AddCall(JsonElement el, List<ParsedCall> res)
    {
        if (el.ValueKind != JsonValueKind.Object)
        {
            return false;
        }

        if (el.TryGetProperty("function", out var fn) && fn.ValueKind == JsonValueKind.Object)
        {
            el = fn;
        }

        if (!el.TryGetProperty("name", out var n) || n.ValueKind != JsonValueKind.String)
        {
            return false;
        }

        string args = "{}";
        foreach (var key in (ReadOnlySpan<string>)["arguments", "parameters", "input"])
        {
            if (el.TryGetProperty(key, out var a))
            {
                if (a.ValueKind == JsonValueKind.String)
                {
                    var s = a.GetString()!;
                    try { using var d = JsonDocument.Parse(s); args = s; } catch (JsonException) { args = JsonString(s); }
                }
                else
                {
                    args = a.GetRawText();
                }

                break;
            }
        }

        res.Add(new ParsedCall(n.GetString()!, args));
        return true;
    }

    /// <summary>Splits text into top-level JSON values (balanced {..} / [..]) ignoring separators between them.</summary>
    private static IEnumerable<string> SplitTopLevel(string s)
    {
        int i = 0;
        while (i < s.Length)
        {
            while (i < s.Length && (char.IsWhiteSpace(s[i]) || s[i] == ';' || s[i] == ','))
            {
                i++;
            }

            if (i >= s.Length)
            {
                yield break;
            }

            if (s[i] != '{' && s[i] != '[')
            {
                yield return s[i..];
                yield break;
            }
            int depth = 0, start = i;
            bool inStr = false;
            for (; i < s.Length; i++)
            {
                char c = s[i];
                if (inStr)
                {
                    if (c == '\\') { i++; } else if (c == '"') { inStr = false; }
                    continue;
                }
                if (c == '"')
                {
                    inStr = true;
                }
                else if (c == '{' || c == '[')
                {
                    depth++;
                }
                else if (c == '}' || c == ']') { if (--depth == 0) { i++; break; } }
            }
            yield return s[start..i];
        }
    }

    private sealed partial class HermesFormat : IToolCallFormat
    {
        public string Name => "hermes";
        public string? StartMarker => "<tool_call>";
        public string? EndMarker => "</tool_call>";
        public bool AllowLeadingJson => false;

        [GeneratedRegex(@"<function=([^>\s]+)>")] private static partial Regex FunctionRx();
        [GeneratedRegex(@"<parameter=([^>\s]+)>\s*?\n?(.*?)\n?</parameter>", RegexOptions.Singleline)] private static partial Regex ParamRx();

        public bool TryParse(string block, out IReadOnlyList<ParsedCall> calls)
        {
            block = block.Trim();
            if (block.StartsWith('{') || block.StartsWith('['))
            {
                return TryParseJsonCalls(block, out calls);
            }
            // Qwen3-Coder / Qwen3.5 style: <function=name><parameter=key>value</parameter></function>
            var res = new List<ParsedCall>();
            calls = res;
            var fm = FunctionRx().Match(block);
            if (!fm.Success)
            {
                return false;
            }

            var sb = new StringBuilder("{");
            bool first = true;
            foreach (Match pm in ParamRx().Matches(block))
            {
                if (!first)
                {
                    sb.Append(',');
                }

                first = false;
                sb.Append(ToolCallFormats.JsonString(pm.Groups[1].Value)).Append(':').Append(ParamValueJson(pm.Groups[2].Value.Trim('\n', '\r')));
            }
            sb.Append('}');
            res.Add(new ParsedCall(fm.Groups[1].Value, sb.ToString()));
            return true;
        }

        private static string ParamValueJson(string v)
        {
            var t = v.Trim();
            if (t.Length > 0)
            {
                try { using var d = JsonDocument.Parse(t); if (d.RootElement.ValueKind != JsonValueKind.String) { return t; } } catch (JsonException) { }
            }

            return ToolCallFormats.JsonString(v);
        }
    }

    private sealed class MistralFormat : IToolCallFormat
    {
        public string Name => "mistral";
        public string? StartMarker => "[TOOL_CALLS]";
        public string? EndMarker => null;
        public bool AllowLeadingJson => false;
        public bool TryParse(string block, out IReadOnlyList<ParsedCall> calls) => TryParseJsonCalls(block, out calls);
    }

    private sealed class Llama3Format : IToolCallFormat
    {
        public string Name => "llama3";
        public string? StartMarker => "<|python_tag|>";
        public string? EndMarker => null;
        public bool AllowLeadingJson => true;
        public bool TryParse(string block, out IReadOnlyList<ParsedCall> calls) => TryParseJsonCalls(block, out calls);
    }
}
