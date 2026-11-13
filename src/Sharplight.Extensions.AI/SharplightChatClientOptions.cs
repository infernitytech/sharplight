using Microsoft.Extensions.AI;
using Sharplight.Llm;
using Sharplight.Vision;

namespace Sharplight.Extensions.AI;

/// <summary>
/// Configuration of a <see cref="SharplightChatClient"/>: default sampling, reasoning mode, tool-call markers and parsers,
/// image decoding and extra chat-template variables. Instances are plain mutable bags; set the properties before handing the
/// object to the client (the client reads them per request and does not copy the collections).
/// </summary>
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
    /// <summary>Text that opens a tool call in the model output (the model's tool-call tag).</summary>
    public string ToolCallStart { get; set; } = "<tool_call>";
    /// <summary>Text that closes a tool call in the model output.</summary>
    public string ToolCallEnd { get; set; } = "</tool_call>";
    /// <summary>Parsers tried in order on the text of each tool call block; the first that recognizes the block wins. Defaults: Hermes/Qwen3 JSON and Qwen3.5 / Qwen3-Coder XML.</summary>
    public IList<IToolCallParser> ToolCallParsers { get; set; } = [new JsonToolCallParser(), new XmlToolCallParser()];
    /// <summary>Decodes image bytes (PNG/JPEG/...) to RGB. Default uses StbImageSharp.</summary>
    public Func<DataContent, ImageInput> ImageDecoder { get; set; } = ImageDecoding.DecodeWithStb;
    /// <summary>Extra variables passed to the chat template for every request.</summary>
    public IDictionary<string, object?> TemplateVariables { get; set; } = new Dictionary<string, object?>();
}
