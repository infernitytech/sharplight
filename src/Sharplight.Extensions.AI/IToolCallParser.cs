using Microsoft.Extensions.AI;

namespace Sharplight.Extensions.AI;

/// <summary>Turns tool-call text emitted by the model into a <see cref="FunctionCallContent"/>.</summary>
public interface IToolCallParser
{
    /// <summary>Parses the text between the tool-call start and end markers.</summary>
    /// <param name="body">Raw text between <see cref="SharplightChatClientOptions.ToolCallStart"/> and <see cref="SharplightChatClientOptions.ToolCallEnd"/>.</param>
    /// <param name="tools">The tools offered in the request, for coercing argument types.</param>
    /// <returns>The call, or null if <paramref name="body"/> is not in this parser's format (the next parser is then tried).</returns>
    public FunctionCallContent? TryParse(string body, ToolSchemas tools);
}
