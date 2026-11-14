using CommunityToolkit.Mvvm.ComponentModel;
using Sharplight.Studio.Models;

namespace Sharplight.Studio.ViewModels.Chat;

/// <summary>A conversation in the sidebar list.</summary>
internal sealed partial class ConversationItemViewModel : ObservableObject
{
    public ConversationItemViewModel(ConversationSummary summary)
    {
        Id = summary.Id;
        Title = summary.Title;
        UpdatedAt = summary.UpdatedAt;
    }

    /// <summary>The conversation id.</summary>
    public string Id { get; }

    /// <summary>Title.</summary>
    [ObservableProperty]
    public partial string Title { get; set; }

    /// <summary>Last change.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(When))]
    public partial DateTimeOffset UpdatedAt { get; set; }

    /// <summary>Renaming in place.</summary>
    [ObservableProperty]
    public partial bool IsRenaming { get; set; }

    /// <summary>An answer is being generated in this conversation (possibly in the background).</summary>
    [ObservableProperty]
    public partial bool IsGenerating { get; set; }

    /// <summary>Relative time of the last change.</summary>
    public string When => RelativeTime.Format(UpdatedAt, DateTimeOffset.UtcNow);
}
