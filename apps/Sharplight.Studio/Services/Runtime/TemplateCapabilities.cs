using Sharplight.Studio.Models;

namespace Sharplight.Studio.Services.Runtime;

/// <summary>Reads what a chat template supports from its source.</summary>
internal static class TemplateCapabilities
{
    /// <summary>Chat, reasoning and tool support of a model with this Jinja template (None without one).</summary>
    public static ModelCapabilities From(string? template)
    {
        if (string.IsNullOrWhiteSpace(template))
        {
            return ModelCapabilities.None;
        }

        var capabilities = ModelCapabilities.Chat;
        if (template.Contains("<think>", StringComparison.Ordinal) || template.Contains("enable_thinking", StringComparison.Ordinal)
            || template.Contains("reasoning_content", StringComparison.Ordinal) || template.Contains("<|channel|>", StringComparison.Ordinal))
        {
            capabilities |= ModelCapabilities.Reasoning;
        }

        if (template.Contains("tools", StringComparison.Ordinal))
        {
            capabilities |= ModelCapabilities.Tools;
        }

        return capabilities;
    }
}
