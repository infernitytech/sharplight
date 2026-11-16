using System.Text;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Runtime;

namespace Sharplight.Studio.ViewModels.Chat;

/// <summary>
/// The streamed text of an answer that the UI has not shown yet: the stream is read on a worker thread and its pieces collect here,
/// and the UI thread appends them to the message in one step per throttled update (re-rendering the markdown once per batch, not once
/// per token).
/// </summary>
internal sealed class ChatStreamBuffer
{
    private readonly Lock _lock = new();
    private readonly StringBuilder _content = new();
    private readonly StringBuilder _reasoning = new();

    /// <summary>The statistics of the finished answer, or null.</summary>
    public MessageStats? Stats { get; private set; }

    /// <summary>The error the answer finished with, or null.</summary>
    public string? Error { get; private set; }

    /// <summary>Records one update of the stream (any thread).</summary>
    /// <param name="update">The update.</param>
    public void Add(ChatStreamUpdate update)
    {
        lock (_lock)
        {
            switch (update.Kind)
            {
                case ChatStreamKind.Content:
                    _content.Append(update.Text);
                    break;
                case ChatStreamKind.Reasoning:
                    _reasoning.Append(update.Text);
                    break;
                case ChatStreamKind.ToolCall:
                    _content.Append($"\n\n`→ {update.Text}`\n");
                    break;
                case ChatStreamKind.Finished:
                    Stats = update.Stats;
                    Error = update.Error;
                    break;
            }
        }
    }

    /// <summary>Appends the text collected since the last call to <paramref name="message"/> (on the UI thread).</summary>
    /// <param name="message">The streaming message.</param>
    public void ApplyTo(MessageViewModel message)
    {
        string content;
        string reasoning;
        lock (_lock)
        {
            content = _content.ToString();
            reasoning = _reasoning.ToString();
            _content.Clear();
            _reasoning.Clear();
        }

        if (reasoning.Length > 0)
        {
            message.AppendReasoning(reasoning);
        }

        if (content.Length > 0)
        {
            message.AppendContent(content);
        }
    }
}
