using Microsoft.Extensions.AI;
using Sharplight.Llm;
using Sharplight.Vision;

namespace Sharplight.Extensions.AI;

public sealed class SharplightChatClientOptions
{
    /// <summary>Sampling used when the request's <see cref="ChatOptions"/> don't override a value.</summary>
    public SamplingOptions Sampling { get; set; } = new();
    /// <summary>Default for <see cref="ChatOptions.MaxOutputTokens"/> (clamped to the remaining context).</summary>
    public int DefaultMaxOutputTokens { get; set; } = 1024;
    /// <summary>
    /// Passes <c>enable_thinking</c> to the chat template (models with a reasoning mode, e.g. Qwen3/3.5). Per request override:
    /// <c>ChatOptions.AdditionalProperties["enable_thinking"]</c>. Reasoning text is returned as <see cref="TextReasoningContent"/>.
    /// </summary>
    public bool EnableThinking { get; set; }
    /// <summary>Overrides the model name reported in responses (defaults to the model definition's name).</summary>
    public string? ModelName { get; set; }
    /// <summary>Marker texts delimiting a tool call in the output.</summary>
    public string ToolCallStart { get; set; } = "<tool_call>";
    public string ToolCallEnd { get; set; } = "</tool_call>";
    /// <summary>Tried in order for each tool call block. Defaults: Hermes/Qwen3 JSON and Qwen3.5 / Qwen3-Coder XML.</summary>
    public IList<IToolCallParser> ToolCallParsers { get; set; } = [new JsonToolCallParser(), new XmlToolCallParser()];
    /// <summary>Decodes image bytes (PNG/JPEG/...) to RGB. Default uses StbImageSharp.</summary>
    public Func<DataContent, ImageInput> ImageDecoder { get; set; } = ImageDecoding.DecodeWithStb;
    /// <summary>Extra variables passed to the chat template for every request.</summary>
    public IDictionary<string, object?> TemplateVariables { get; set; } = new Dictionary<string, object?>();
}

static class ImageDecoding
{
    public static ImageInput DecodeWithStb(DataContent content)
    {
        var img = StbImageSharp.ImageResult.FromMemory(content.Data.ToArray(), StbImageSharp.ColorComponents.RedGreenBlue);
        return new ImageInput(img.Width, img.Height, img.Data);
    }
}
