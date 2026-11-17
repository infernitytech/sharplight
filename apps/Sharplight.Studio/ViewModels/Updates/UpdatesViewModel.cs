using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Studio.Services;
using Sharplight.Studio.Services.Updates;

namespace Sharplight.Studio.ViewModels.Updates;

/// <summary>
/// Update state shared by the "update available" dialog and the Settings section: the automatic and manual checks, the offer with its
/// release notes, the download and the restart that installs it.
/// </summary>
internal sealed partial class UpdatesViewModel : ObservableObject, IDisposable
{
    private static readonly TimeSpan s_interval = TimeSpan.FromHours(6);

    private readonly StudioServices _services;
    private readonly TimeProvider _time;
    private readonly TimeSpan _interval;
    private readonly CancellationTokenSource _lifetime = new();
    private bool _initializing = true;
    private bool _started;
    private string? _declinedDowngrade;

    public UpdatesViewModel(StudioServices services, TimeProvider? time = null, TimeSpan? interval = null)
    {
        _services = services;
        _time = time ?? TimeProvider.System;
        _interval = interval ?? s_interval;
        AutoCheck = services.Settings.Current.AutoCheckForUpdates;
        LastChecked = services.Settings.Current.LastUpdateCheck;
        StatusText = services.Updates.Support.IsSupported ? "" : services.Updates.Support.Reason ?? "";
        InstalledKind = UpdateChannels.KindOf(services.Updates.Channel);
        SelectedChannel = UpdateChannelChoice.For(UpdateChannels.Parse(services.Settings.Current.UpdateChannel) ?? InstalledKind);
        _initializing = false;
    }

    /// <summary>The running version.</summary>
    public string CurrentVersion => _services.Updates.CurrentVersion;

    /// <summary>The stream of the channel this install came from.</summary>
    public UpdateChannelKind InstalledKind { get; }

    /// <summary>The installed channel for display ("Stable", "Preview (unstable)", or a note when not installed).</summary>
    public string InstalledChannelText => _services.Updates.Channel is null ? "none (not installed)" : UpdateChannelChoice.For(InstalledKind).Title;

    /// <summary>The channels to pick from.</summary>
    public IReadOnlyList<UpdateChannelChoice> Channels => UpdateChannelChoice.All;

    /// <summary>The channel updates are checked against; saved in the settings (a choice equal to the installed one is kept explicitly).</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(SelectedChannelName), nameof(NoReleaseText))]
    public partial UpdateChannelChoice SelectedChannel { get; set; }

    /// <summary>The feed name of the selected channel, e.g. <c>linux-x64-stable</c>; null when not installed.</summary>
    public string? SelectedChannelName => UpdateChannels.Name(_services.Updates.Channel, SelectedChannel.Kind);

    /// <summary>The channel can be switched (updates supported, nothing running).</summary>
    public bool CanSwitchChannel => IsSupported && !IsChecking && !IsDownloading;

    /// <summary>The selected channel has no release yet.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(NoReleaseText))]
    public partial bool HasNoRelease { get; set; }

    /// <summary>"No stable release yet" for the selected channel.</summary>
    public string NoReleaseText => $"No {ChannelWord} release yet: nothing is published on this channel so far. Studio stays on the current version.";

    private string ChannelWord => SelectedChannel.Kind == UpdateChannelKind.Preview ? "preview" : "stable";

    /// <summary>The selected channel is behind the running version: its latest release, waiting for the user's decision; null when none.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasDowngradeOffer), nameof(DowngradeText))]
    public partial UpdateOffer? DowngradeOffer { get; set; }

    /// <summary>Installing the channel's release would downgrade Studio.</summary>
    public bool HasDowngradeOffer => DowngradeOffer is not null;

    /// <summary>The question shown before a downgrade.</summary>
    public string DowngradeText => DowngradeOffer is { } offer
        ? $"The {SelectedChannel.Title} channel is at {offer.Version}, which is older than the {CurrentVersion} you are running. "
            + "Installing it is a downgrade and may not read data written by the newer version. Downgrade, or stay on the current version until this channel catches up?"
        : "";

    /// <summary>Updates can be checked for (installed, feed configured).</summary>
    public bool IsSupported => _services.Updates.Support.IsSupported;

    /// <summary>Why updates are unavailable; empty when they are available.</summary>
    public string UnavailableReason => _services.Updates.Support.Reason ?? "";

    /// <summary>Check at startup and every few hours.</summary>
    [ObservableProperty]
    public partial bool AutoCheck { get; set; }

    /// <summary>When the last check finished; null when none ran.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(LastCheckedText))]
    public partial DateTimeOffset? LastChecked { get; set; }

    /// <summary>The last check time for display ("never", "5 min ago").</summary>
    public string LastCheckedText => LastChecked is { } at ? RelativeTime.Format(at, _time.GetUtcNow()) : "never";

    /// <summary>Result of the last check or action, shown under the buttons.</summary>
    [ObservableProperty]
    public partial string StatusText { get; set; }

    /// <summary>A check is running.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanSwitchChannel))]
    [NotifyCanExecuteChangedFor(nameof(CheckNowCommand))]
    public partial bool IsChecking { get; set; }

    /// <summary>The newer version found by the last check; null when none.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasOffer), nameof(OfferTitle), nameof(ReleaseNotes), nameof(HasReleaseNotes), nameof(CanDownload))]
    [NotifyCanExecuteChangedFor(nameof(DownloadCommand))]
    public partial UpdateOffer? Offer { get; set; }

    /// <summary>The "update available" dialog is open.</summary>
    [ObservableProperty]
    public partial bool IsDialogOpen { get; set; }

    /// <summary>The package is downloading.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanDownload), nameof(CanSwitchChannel))]
    [NotifyCanExecuteChangedFor(nameof(DownloadCommand), nameof(CheckNowCommand))]
    public partial bool IsDownloading { get; set; }

    /// <summary>Download progress, 0 to 100.</summary>
    [ObservableProperty]
    public partial int DownloadPercent { get; set; }

    /// <summary>The package is downloaded; restarting installs it.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanDownload))]
    [NotifyCanExecuteChangedFor(nameof(RestartCommand), nameof(DownloadCommand))]
    public partial bool IsReady { get; set; }

    /// <summary>The download failed or the install could not start.</summary>
    [ObservableProperty]
    public partial string? Error { get; set; }

    /// <summary>A newer version was found.</summary>
    public bool HasOffer => Offer is not null;

    /// <summary>The download button is offered.</summary>
    public bool CanDownload => HasOffer && !IsDownloading && !IsReady;

    /// <summary>Dialog title text: "Sharplight Studio 0.2.7 is available".</summary>
    public string OfferTitle => Offer is { IsDowngrade: true } older ? $"Sharplight Studio {older.Version} (downgrade from {CurrentVersion})"
        : Offer is { } offer ? $"Sharplight Studio {offer.Version} is available (you have {CurrentVersion})" : "";

    /// <summary>The Markdown release notes of the offer, or a placeholder.</summary>
    public string ReleaseNotes => Offer?.ReleaseNotes is { Length: > 0 } notes ? notes : "_No release notes._";

    /// <summary>The offer has release notes.</summary>
    public bool HasReleaseNotes => Offer is not null;

    /// <summary>
    /// Starts the automatic checks: one now and one every few hours while <see cref="AutoCheck"/> is on. Does nothing when updates are
    /// unsupported (dev builds, no feed URL) or when called twice. Call it on the UI thread and do not await it: the task ends when the
    /// view model is disposed.
    /// </summary>
    public Task StartAsync()
    {
        if (_started || !IsSupported)
        {
            return Task.CompletedTask;
        }

        _started = true;
        return RunAsync(_lifetime.Token);
    }

    /// <summary>Runs one check; <paramref name="manual"/> checks report "up to date" and open the dialog for an offer.</summary>
    public async Task CheckAsync(bool manual)
    {
        if (!IsSupported)
        {
            StatusText = UnavailableReason;
            return;
        }

        if (IsChecking || IsDownloading)
        {
            return;
        }

        IsChecking = true;
        Error = null;
        try
        {
            var result = await _services.Updates.CheckAsync(SelectedChannelName, _lifetime.Token);
            Apply(result, manual);
        }
        catch (OperationCanceledException)
        {
            // Closing.
        }
        finally
        {
            IsChecking = false;
        }
    }

    /// <summary>Releases the periodic check.</summary>
    public void Dispose()
    {
        _lifetime.Cancel();
        _lifetime.Dispose();
    }

    partial void OnAutoCheckChanged(bool value)
    {
        if (!_initializing)
        {
            _services.Settings.Update(settings => settings with { AutoCheckForUpdates = value });
        }
    }

    partial void OnSelectedChannelChanged(UpdateChannelChoice value)
    {
        if (_initializing)
        {
            return;
        }

        _services.Settings.Update(settings => settings with { UpdateChannel = UpdateChannels.Key(value.Kind) });
        Offer = null;
        DowngradeOffer = null;
        IsReady = false;
        HasNoRelease = false;
        _declinedDowngrade = null;
        if (IsSupported)
        {
            _ = CheckAsync(manual: true);
        }
    }

    private void Apply(UpdateCheckResult result, bool manual)
    {
        LastChecked = _time.GetUtcNow();
        _services.Settings.Update(settings => settings with { LastUpdateCheck = LastChecked });
        if (result.Error is { } error)
        {
            StatusText = manual ? $"Update check failed: {error}" : StatusText;
            return;
        }

        HasNoRelease = result.NoRelease;
        if (result.NoRelease)
        {
            Offer = null;
            DowngradeOffer = null;
            StatusText = $"No {ChannelWord} release yet. Studio {CurrentVersion} stays as it is.";
            return;
        }

        if (result.Offer is { IsDowngrade: true } older)
        {
            Offer = null;
            if (older.Version == _declinedDowngrade)
            {
                DowngradeOffer = null;
                StatusText = $"Staying on {CurrentVersion} until the {SelectedChannel.Title} channel catches up (it is at {older.Version}).";
            }
            else
            {
                DowngradeOffer = older;
                StatusText = $"The {SelectedChannel.Title} channel is at {older.Version}, older than {CurrentVersion}.";
            }

            return;
        }

        DowngradeOffer = null;
        if (result.Offer is { } offer)
        {
            bool isNew = Offer?.Version != offer.Version;
            Offer = offer;
            IsReady = IsReady && !isNew;
            StatusText = $"Version {offer.Version} is available.";
            if (manual || isNew)
            {
                IsDialogOpen = true;
            }

            return;
        }

        Offer = null;
        StatusText = $"Studio {CurrentVersion} is up to date.";
    }

    private async Task RunAsync(CancellationToken cancellationToken)
    {
        using var timer = new PeriodicTimer(_interval, _time);
        try
        {
            do
            {
                if (AutoCheck)
                {
                    await CheckAsync(manual: false);
                }
            }
            while (await timer.WaitForNextTickAsync(cancellationToken));
        }
        catch (OperationCanceledException)
        {
            // Closing.
        }
    }

    [RelayCommand(CanExecute = nameof(CanCheckNow))]
    private Task CheckNowAsync() => CheckAsync(manual: true);

    private bool CanCheckNow() => !IsChecking && !IsDownloading;

    [RelayCommand(CanExecute = nameof(CanDownload))]
    private async Task DownloadAsync()
    {
        if (Offer is not { } offer)
        {
            return;
        }

        Error = null;
        DownloadPercent = 0;
        IsDownloading = true;
        try
        {
            var progress = new Progress<int>(percent => DownloadPercent = Math.Max(DownloadPercent, percent));
            await _services.Updates.DownloadAsync(offer, progress, _lifetime.Token);
            DownloadPercent = 100;
            IsReady = true;
            StatusText = $"Version {offer.Version} is ready: restart Studio to install it.";
        }
        catch (OperationCanceledException)
        {
            // Closing.
        }
        catch (Exception error)
        {
            Error = $"The download failed: {error.Message}";
        }
        finally
        {
            IsDownloading = false;
        }
    }

    [RelayCommand(CanExecute = nameof(IsReady))]
    private void Restart()
    {
        if (Offer is not { } offer)
        {
            return;
        }

        try
        {
            _services.Updates.ApplyAndRestart(offer);
        }
        catch (Exception error)
        {
            Error = $"The update could not be installed: {error.Message}";
        }
    }

    /// <summary>Accepts the downgrade: the older release becomes the offer, which is then downloaded and installed like any update.</summary>
    [RelayCommand]
    private void ConfirmDowngrade()
    {
        if (DowngradeOffer is not { } older)
        {
            return;
        }

        _declinedDowngrade = null;
        DowngradeOffer = null;
        Offer = older;
        IsReady = false;
        StatusText = $"Version {older.Version} (a downgrade) is ready to download.";
    }

    /// <summary>Declines the downgrade and keeps the running version; the question is not asked again for that release.</summary>
    [RelayCommand]
    private void StayOnCurrent()
    {
        if (DowngradeOffer is not { } older)
        {
            return;
        }

        _declinedDowngrade = older.Version;
        DowngradeOffer = null;
        StatusText = $"Staying on {CurrentVersion} until the {SelectedChannel.Title} channel catches up (it is at {older.Version}).";
    }

    /// <summary>Closes the dialog; the offer stays visible in Settings.</summary>
    [RelayCommand]
    private void Later() => IsDialogOpen = false;
}
