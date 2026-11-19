using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Studio.Services;
using Sharplight.Studio.Services.Library;
using Sharplight.Studio.Services.Runtime;

namespace Sharplight.Studio.ViewModels.Inference;

/// <summary>
/// The licence callout of the chosen model: shown when the model came from a catalog entry with a non-permissive licence, until the user dismisses it;
/// the dismissal is stored in the settings per catalog entry, so the notice appears once per model and not at every selection.
/// </summary>
internal sealed partial class LicenseNoticeViewModel : ObservableObject
{
    private readonly StudioServices _services;
    private CancellationTokenSource? _lookup;
    private string _entryId = "";

    public LicenseNoticeViewModel(StudioServices services)
    {
        _services = services;
    }

    /// <summary>The notice, or empty.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsVisible))]
    public partial string Text { get; set; } = "";

    /// <summary>There is a notice the user has not dismissed.</summary>
    public bool IsVisible => Text.Length > 0;

    /// <summary>Looks up the notice of <paramref name="model"/> (off the UI thread) and shows it unless it was dismissed before.</summary>
    /// <param name="model">The chosen model, or null.</param>
    public async Task RefreshAsync(ILoadedModel? model)
    {
        _lookup?.Cancel();
        _lookup = new CancellationTokenSource();
        var token = _lookup.Token;
        if (model is null)
        {
            Text = "";
            return;
        }

        var notice = await LicenseNotices.FindAsync(_services, model.Path, token).ConfigureAwait(true);
        if (token.IsCancellationRequested)
        {
            return;
        }

        _entryId = notice?.EntryId ?? "";
        Text = notice is null || LicenseNotices.IsDismissed(_services, notice.EntryId) ? "" : notice.Text;
    }

    /// <summary>Hides the notice for good: this model's catalog entry is remembered in the settings.</summary>
    [RelayCommand]
    private void Dismiss()
    {
        if (_entryId.Length > 0)
        {
            LicenseNotices.Dismiss(_services, _entryId);
        }

        Text = "";
    }
}
