using System.Text.Json;
using Microsoft.Extensions.AI;
using Sharplight.Llm.Serving.Chat;

namespace Sharplight.Extensions.AI;

/// <summary>Maps the shared <see cref="ChatOutputParser"/>'s pieces to Microsoft.Extensions.AI contents (both chat clients use it).</summary>
internal static class ChatOutputContents
{
    /// <summary>The content of a parsed piece: text, reasoning or a function call.</summary>
    public static AIContent From(ChatOutputPiece piece) => piece.Kind switch
    {
        ChatOutputPieceKind.Content => new TextContent(piece.Text),
        ChatOutputPieceKind.Reasoning => new TextReasoningContent(piece.Text),
        _ => From(piece.Call!),
    };

    /// <summary>A function call with its arguments: JSON strings become strings, other values <see cref="JsonElement"/>s.</summary>
    public static FunctionCallContent From(ToolCall call) => new(call.Id, call.Name, ParseArguments(call.ArgumentsJson));

    private static Dictionary<string, object?> ParseArguments(string json)
    {
        var arguments = new Dictionary<string, object?>(StringComparer.Ordinal);
        try
        {
            using var document = JsonDocument.Parse(json);
            if (document.RootElement.ValueKind == JsonValueKind.Object)
            {
                foreach (var property in document.RootElement.EnumerateObject())
                {
                    arguments[property.Name] = property.Value.ValueKind == JsonValueKind.String ? property.Value.GetString() : property.Value.Clone();
                }
            }
        }
        catch (JsonException)
        {
            // Unparsable arguments arrive as an empty set.
        }

        return arguments;
    }
}
