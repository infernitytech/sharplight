using System.Text;
using System.Text.Json;

namespace Sharplight.HuggingFace;

/// <summary>Reads a checkpoint's Jinja chat template from <c>tokenizer_config.json</c> or the dedicated template files.</summary>
internal static class HfChatTemplate
{
    /// <summary>
    /// The chat template, or null when the checkpoint has none. A <c>chat_template.jinja</c> file wins over
    /// <c>chat_template.json</c>, which wins over the <c>chat_template</c> entry of tokenizer_config.json.
    /// </summary>
    /// <param name="dir">The checkpoint directory.</param>
    /// <param name="tokenizerConfig">The root of tokenizer_config.json.</param>
    public static string? Read(string dir, JsonElement tokenizerConfig)
    {
        string? template = null;
        if (tokenizerConfig.TryGetProperty("chat_template", out var inline))
        {
            template = TemplateFrom(inline);
        }

        string jinjaPath = Path.Combine(dir, "chat_template.jinja");
        string jsonPath = Path.Combine(dir, "chat_template.json");
        if (File.Exists(jinjaPath))
        {
            template = File.ReadAllText(jinjaPath, Encoding.UTF8);
        }
        else if (File.Exists(jsonPath))
        {
            using var document = JsonDocument.Parse(File.ReadAllBytes(jsonPath));
            if (document.RootElement.TryGetProperty("chat_template", out var fromFile))
            {
                template = TemplateFrom(fromFile);
            }
        }

        return template;
    }

    /// <summary>A string template, or a list of named templates collapsed to the "default" (else first) one.</summary>
    private static string? TemplateFrom(JsonElement element)
    {
        if (element.ValueKind == JsonValueKind.String)
        {
            return element.GetString();
        }

        if (element.ValueKind != JsonValueKind.Array)
        {
            return null;
        }

        string? first = null;
        foreach (var item in element.EnumerateArray())
        {
            if (!item.TryGetProperty("template", out var template) || template.ValueKind != JsonValueKind.String)
            {
                continue;
            }

            first ??= template.GetString();
            if (item.TryGetProperty("name", out var name) && name.GetString() == "default")
            {
                return template.GetString();
            }
        }

        return first;
    }
}
