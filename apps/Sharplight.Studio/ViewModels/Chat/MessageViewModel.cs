using System.Collections.ObjectModel;
using System.Globalization;
using System.Text;
using CommunityToolkit.Mvvm.ComponentModel;
using Sharplight.Studio.Models;

namespace Sharplight.Studio.ViewModels.Chat;

/// <summary>A message in the chat, streaming while it is generated.</summary>
internal sealed partial class MessageViewModel : ObservableObject
{
    private readonly StringBuilder _content = new();
    private readonly StringBuilder _reasoning = new();

    public MessageViewModel(ChatMessageRecord record, int versionIndex, int versionCount)
    {
        Record = record;
        _content.Append(record.Content);
        _reasoning.Append(record.Reasoning);
        Content = record.Content;
        Reasoning = record.Reasoning ?? "";
        Error = record.Error;
        VersionIndex = versionIndex;
        VersionCount = versionCount;
        Attachments = new ObservableCollection<AttachmentViewModel>(record.Attachments.Select(attachment => new AttachmentViewModel(attachment)));
        Stats = record.Stats;
    }

    /// <summary>The stored message (updated when generation finishes).</summary>
    public ChatMessageRecord Record { get; private set; }

    /// <summary>Message id.</summary>
    public string Id => Record.Id;

    /// <summary>True for the user's messages.</summary>
    public bool IsUser => Record.Role == ChatRole.User;

    /// <summary>True for the model's answers.</summary>
    public bool IsAssistant => Record.Role == ChatRole.Assistant;

    /// <summary>The model that answered.</summary>
    public string ModelName => Record.ModelId ?? "Assistant";

    /// <summary>Attachments.</summary>
    public ObservableCollection<AttachmentViewModel> Attachments { get; }

    /// <summary>True when there are attachments.</summary>
    public bool HasAttachments => Attachments.Count > 0;

    /// <summary>Text (Markdown).</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsThinking))]
    public partial string Content { get; set; }

    /// <summary>Reasoning text.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasReasoning), nameof(IsThinking), nameof(ReasoningHeader))]
    public partial string Reasoning { get; set; }

    /// <summary>The reasoning section is open.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(ReasoningHeader))]
    public partial bool IsReasoningExpanded { get; set; }

    /// <summary>Still generating.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsThinking), nameof(ShowActions), nameof(ReasoningHeader), nameof(IsWaiting))]
    public partial bool IsStreaming { get; set; }

    /// <summary>Failure message.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasError))]
    public partial string? Error { get; set; }

    /// <summary>Statistics of the answer.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(StatsText), nameof(HasStats), nameof(StatsTooltip))]
    public partial MessageStats? Stats { get; set; }

    /// <summary>1-based position among the message's versions.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasVersions), nameof(VersionText), nameof(CanShowPrevious), nameof(CanShowNext))]
    public partial int VersionIndex { get; set; }

    /// <summary>Number of versions (edits or regenerations).</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasVersions), nameof(VersionText), nameof(CanShowPrevious), nameof(CanShowNext))]
    public partial int VersionCount { get; set; }

    /// <summary>Being edited in place.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(ShowActions))]
    public partial bool IsEditing { get; set; }

    /// <summary>The text while editing.</summary>
    [ObservableProperty]
    public partial string EditText { get; set; } = "";

    /// <summary>True when there is reasoning.</summary>
    public bool HasReasoning => Reasoning.Length > 0;

    /// <summary>Reasoning is streaming and the answer has not started yet.</summary>
    public bool IsThinking => IsStreaming && Content.Length == 0 && Reasoning.Length > 0;

    /// <summary>Generating but nothing arrived yet.</summary>
    public bool IsWaiting => IsStreaming && Content.Length == 0 && Reasoning.Length == 0;

    /// <summary>Title of the reasoning section.</summary>
    public string ReasoningHeader => IsThinking ? "Thinking…" : IsReasoningExpanded ? "Hide reasoning" : "Show reasoning";

    /// <summary>True when there is an error.</summary>
    public bool HasError => !string.IsNullOrEmpty(Error);

    /// <summary>Show copy / regenerate / edit buttons.</summary>
    public bool ShowActions => !IsStreaming && !IsEditing;

    /// <summary>True when statistics are known.</summary>
    public bool HasStats => Stats is { CompletionTokens: > 0 };

    /// <summary>One-line statistics: decode speed, tokens, time to first token.</summary>
    public string StatsText
    {
        get
        {
            if (Stats is not { } s)
            {
                return "";
            }

            var text = string.Create(CultureInfo.InvariantCulture, $"{s.DecodeTokensPerSecond:F1} tok/s · {s.CompletionTokens} tokens · {s.TimeToFirstTokenMs / 1000:F2}s to first token");
            if (s.AcceptanceRate is { } rate)
            {
                text += string.Create(CultureInfo.InvariantCulture, $" · {rate:P0} drafts accepted");
            }

            return s.Finish is "stop" or "tool_calls" ? text : $"{text} · {s.Finish}";
        }
    }

    /// <summary>Details for the statistics tooltip.</summary>
    public string StatsTooltip
    {
        get
        {
            if (Stats is not { } s)
            {
                return "";
            }

            return string.Create(CultureInfo.InvariantCulture,
                $"Prompt: {s.PromptTokens} tokens ({s.CachedTokens} cached), prefill {s.PrefillTokensPerSecond:F0} tok/s\nQueue {s.QueueMs:F0} ms · prefill {s.PrefillMs:F0} ms · decode {s.DecodeMs:F0} ms · total {s.TotalMs:F0} ms");
        }
    }

    /// <summary>True when there are several versions.</summary>
    public bool HasVersions => VersionCount > 1;

    /// <summary><c>2 / 3</c>.</summary>
    public string VersionText => $"{VersionIndex} / {VersionCount}";

    /// <summary>An older version exists.</summary>
    public bool CanShowPrevious => VersionIndex > 1;

    /// <summary>A newer version exists.</summary>
    public bool CanShowNext => VersionIndex < VersionCount;

    /// <summary>Appends streamed answer text.</summary>
    public void AppendContent(string text)
    {
        _content.Append(text);
        Content = _content.ToString();
    }

    /// <summary>Appends streamed reasoning text.</summary>
    public void AppendReasoning(string text)
    {
        _reasoning.Append(text);
        Reasoning = _reasoning.ToString();
    }

    /// <summary>Replaces the stored record (after generation).</summary>
    public void Update(ChatMessageRecord record)
    {
        Record = record;
        Stats = record.Stats;
        Error = record.Error;
    }
}
