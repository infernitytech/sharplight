using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Studio.Services.Library;
using Sharplight.Studio.Services.Platform;

namespace Sharplight.Studio.ViewModels;

/// <summary>
/// The "accept the licence terms" dialog shown before a model with a restrictive licence is downloaded (catalog, Hub, model feed, first run, update all):
/// the licence name, the notice, a link to the full licence, Accept and Cancel. It shows the oldest request of the <see cref="ILicenseGate"/>.
/// </summary>
internal sealed partial class LicenseDialogViewModel : ObservableObject
{
    private readonly ILicenseGate _gate;
    private readonly IPlatformServices _platform;
    private readonly IUiDispatcher _dispatcher;

    public LicenseDialogViewModel(ILicenseGate gate, IPlatformServices platform, IUiDispatcher dispatcher)
    {
        _gate = gate;
        _platform = platform;
        _dispatcher = dispatcher;
        _gate.PendingChanged += (_, _) => _dispatcher.Post(Refresh);
        Refresh();
    }

    /// <summary>A request is waiting for the user's answer.</summary>
    [ObservableProperty]
    public partial bool IsOpen { get; private set; }

    /// <summary>The model's name.</summary>
    [ObservableProperty]
    public partial string ModelName { get; private set; } = "";

    /// <summary>The licence name.</summary>
    [ObservableProperty]
    public partial string License { get; private set; } = "";

    /// <summary>The notice text.</summary>
    [ObservableProperty]
    public partial string Notice { get; private set; } = "";

    /// <summary>The link to the full licence, or empty.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasLink))]
    public partial string Link { get; private set; } = "";

    /// <summary>There is a link to the full licence.</summary>
    public bool HasLink => Link.Length > 0;

    /// <summary>The title line.</summary>
    public string Title => $"Accept the licence terms of {ModelName}";

    /// <summary>Accepts the terms (remembered for this model); the download goes on.</summary>
    [RelayCommand]
    private void Accept() => _gate.Accept();

    /// <summary>Declines: nothing is downloaded.</summary>
    [RelayCommand]
    private void Cancel() => _gate.Decline();

    /// <summary>Opens the full licence in the browser.</summary>
    [RelayCommand]
    private void OpenLicense()
    {
        if (HasLink)
        {
            _platform.Open(Link);
        }
    }

    private void Refresh()
    {
        var terms = _gate.Pending;
        ModelName = terms?.ModelName ?? "";
        License = terms?.License ?? "";
        Notice = terms?.Notice ?? "";
        Link = terms?.Url ?? "";
        OnPropertyChanged(nameof(Title));
        IsOpen = terms is not null;
    }
}
