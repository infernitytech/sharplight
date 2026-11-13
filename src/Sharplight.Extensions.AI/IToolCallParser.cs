using Microsoft.Extensions.AI;

namespace Sharplight.Extensions.AI;

/// <summary>Turns tool-call text emitted by the model into a <see cref="FunctionCallContent"/>.</summary>
public interface IToolCallParser
{
    /// <summary>Parses the text between the tool-call start and end markers; returns null if it is not this parser's format.</summary>
    public FunctionCallContent? TryParse(string body, ToolSchemas tools);
}
