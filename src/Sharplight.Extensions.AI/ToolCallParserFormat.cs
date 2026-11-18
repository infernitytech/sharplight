using System.Text.Json;
using Microsoft.Extensions.AI;
using Sharplight.Llm.Serving.Chat;

namespace Sharplight.Extensions.AI;

/// <summary>
/// The tool-call format of a <see cref="SharplightChatClient"/> request for the shared <see cref="ChatOutputParser"/>: blocks between
/// <see cref="SharplightChatClientOptions.ToolCallStart"/> and <see cref="SharplightChatClientOptions.ToolCallEnd"/>, read by the
/// configured <see cref="IToolCallParser"/>s (which coerce arguments to the request's tool schemas).
/// </summary>
internal sealed class ToolCallParserFormat : IToolCallFormat
{
    private readonly IList<IToolCallParser> _parsers;
    private readonly ToolSchemas _tools;

    public ToolCallParserFormat(string startMarker, string endMarker, IList<IToolCallParser> parsers, ToolSchemas tools)
    {
        StartMarker = startMarker;
        EndMarker = endMarker;
        _parsers = parsers;
        _tools = tools;
    }

    public string Name => "sharplight-chat-client";

    public string? StartMarker { get; }

    public string? EndMarker { get; }

    public bool AllowLeadingJson => false;

    public bool TryParse(string block, out IReadOnlyList<ParsedCall> calls)
    {
        foreach (var parser in _parsers)
        {
            if (parser.TryParse(block, _tools) is { } call)
            {
                var arguments = call.Arguments ?? new Dictionary<string, object?>();
                string json = JsonSerializer.Serialize(arguments, AIJsonUtilities.DefaultOptions.GetTypeInfo(typeof(IDictionary<string, object?>)));
                calls = [new ParsedCall(call.Name, json)];
                return true;
            }
        }

        calls = [];
        return false;
    }
}
