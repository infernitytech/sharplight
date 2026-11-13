using System.Text.Json.Nodes;

namespace Sharplight.AspNetCore;

/// <summary>
/// Writes the content blocks of one streamed Anthropic message: consecutive text or thinking deltas share a block, a tool call is
/// a block of its own, and a message that produced nothing still gets an empty text block, as Anthropic's API does.
/// </summary>
internal sealed class AnthropicBlockWriter(SseWriter sse)
{
    private const string TextType = "text";
    private const string ThinkingType = "thinking";

    private int _index = -1;
    private string? _openType;

    /// <summary>Appends a text delta, opening a text block when another kind is open.</summary>
    public void Text(string text)
    {
        OpenIfNot(TextType);
        Delta("text_delta", "text", text);
    }

    /// <summary>Appends a thinking delta, opening a thinking block when another kind is open.</summary>
    public void Thinking(string thinking)
    {
        OpenIfNot(ThinkingType);
        Delta("thinking_delta", "thinking", thinking);
    }

    /// <summary>Writes a complete tool-use block whose input arrives as one JSON delta.</summary>
    public void ToolUse(string id, string name, string argumentsJson)
    {
        Close();
        _index++;
        sse.Event("content_block_start", new JsonObject
        {
            ["type"] = "content_block_start",
            ["index"] = _index,
            ["content_block"] = new JsonObject { ["type"] = "tool_use", ["id"] = id, ["name"] = name, ["input"] = new JsonObject() },
        });
        Delta("input_json_delta", "partial_json", argumentsJson);
        sse.Event("content_block_stop", new JsonObject { ["type"] = "content_block_stop", ["index"] = _index });
    }

    /// <summary>Closes the open block; opens an empty text block first when the message had no content at all.</summary>
    public void CloseAll()
    {
        if (_index < 0 && _openType is null)
        {
            Open(TextType);
        }

        Close();
    }

    private void OpenIfNot(string type)
    {
        if (_openType != type)
        {
            Open(type);
        }
    }

    private void Open(string type)
    {
        Close();
        _index++;
        _openType = type;
        sse.Event("content_block_start", new JsonObject
        {
            ["type"] = "content_block_start",
            ["index"] = _index,
            ["content_block"] = type == TextType ? new JsonObject { ["type"] = TextType, ["text"] = "" } : new JsonObject { ["type"] = ThinkingType, ["thinking"] = "" },
        });
    }

    private void Close()
    {
        if (_openType is null)
        {
            return;
        }

        if (_openType == ThinkingType)
        {
            sse.Event("content_block_delta", new JsonObject
            {
                ["type"] = "content_block_delta",
                ["index"] = _index,
                ["delta"] = new JsonObject { ["type"] = "signature_delta", ["signature"] = "" },
            });
        }

        sse.Event("content_block_stop", new JsonObject { ["type"] = "content_block_stop", ["index"] = _index });
        _openType = null;
    }

    private void Delta(string kind, string field, string value)
    {
        var w = sse.BeginEvent("content_block_delta");
        w.WriteStartObject();
        w.WriteString("type", "content_block_delta");
        w.WriteNumber("index", _index);
        w.WriteStartObject("delta");
        w.WriteString("type", kind);
        w.WriteString(field, value);
        w.WriteEndObject();
        w.WriteEndObject();
        sse.EndEvent();
    }
}
