using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Updates;
using Sharplight.Studio.ViewModels.Updates;

namespace Sharplight.Studio.Tests;

/// <summary>Switching between the stable and the preview update channel.</summary>
public sealed class UpdateChannelTests
{
    private const string Stable = "linux-x64-stable";
    private const string Preview = "linux-x64-unstable";

    private static UpdatesViewModel Create(TestServices services) => new(services.Services, interval: TimeSpan.FromDays(1));

    [Theory]
    [InlineData("linux-x64-stable", false)]
    [InlineData("linux-x64-unstable", true)]
    [InlineData(null, false)]
    public void Channel_FollowsTheInstalledBuild_ByDefault(string? installed, bool preview)
    {
        using var services = new TestServices();
        services.Updates.Channel = installed;

        using var updates = Create(services);
        var expected = preview ? UpdateChannelKind.Preview : UpdateChannelKind.Stable;

        Assert.Equal(expected, updates.SelectedChannel.Kind);
        Assert.Equal(expected, updates.InstalledKind);
    }

    [Fact]
    public void Names_SwapTheSuffix_AndKeepThePlatform()
    {
        Assert.Equal("win-x64-unstable", UpdateChannels.Name("win-x64-stable", UpdateChannelKind.Preview));
        Assert.Equal("win-x64-stable", UpdateChannels.Name("win-x64-unstable", UpdateChannelKind.Stable));
        Assert.Equal("osx-arm64-stable", UpdateChannels.Name("osx-arm64", UpdateChannelKind.Stable));
        Assert.Null(UpdateChannels.Name(null, UpdateChannelKind.Preview));
    }

    [Fact]
    public void SavedChoice_WinsOverTheInstalledChannel()
    {
        using var services = new TestServices(new AppSettings { UpdateChannel = "unstable" });

        using var updates = Create(services);

        Assert.Equal(UpdateChannelKind.Preview, updates.SelectedChannel.Kind);
        Assert.Equal(Preview, updates.SelectedChannelName);
        Assert.Equal("Stable", updates.InstalledChannelText);
    }

    [Fact]
    public void Switching_PersistsTheChoice_AndChecksTheChosenFeed()
    {
        using var services = new TestServices();
        services.Updates.ResultsByChannel[Preview] = UpdateCheckResult.Found(new UpdateOffer("0.3.0-alpha.1", null, 1));
        using var updates = Create(services);

        updates.SelectedChannel = UpdateChannelChoice.For(UpdateChannelKind.Preview);

        Assert.Equal("unstable", services.Settings.Current.UpdateChannel);
        Assert.Equal([Preview], services.Updates.CheckedChannels);
        Assert.Equal("0.3.0-alpha.1", updates.Offer?.Version);
        Assert.True(updates.DownloadCommand.CanExecute(null));
    }

    [Fact]
    public void Switching_DropsTheOfferOfTheOtherChannel()
    {
        using var services = new TestServices();
        services.Updates.ResultsByChannel[Stable] = UpdateCheckResult.Found(new UpdateOffer("0.2.0", null, 1));
        using var updates = Create(services);
        _ = updates.CheckAsync(manual: false);
        Assert.NotNull(updates.Offer);

        updates.SelectedChannel = UpdateChannelChoice.For(UpdateChannelKind.Preview);

        Assert.Null(updates.Offer);
        Assert.Equal(Preview, services.Updates.CheckedChannels[^1]);
    }

    [Fact]
    public void NoStableRelease_IsAMessage_NotAnError()
    {
        using var services = new TestServices();
        services.Updates.Channel = Preview;
        services.Updates.ResultsByChannel[Stable] = UpdateCheckResult.NoReleaseYet;
        using var updates = Create(services);

        updates.SelectedChannel = UpdateChannelChoice.For(UpdateChannelKind.Stable);

        Assert.True(updates.HasNoRelease);
        Assert.Contains("No stable release yet", updates.NoReleaseText);
        Assert.Contains("No stable release yet", updates.StatusText);
        Assert.Null(updates.Error);
        Assert.False(updates.HasOffer);
        Assert.False(updates.IsDialogOpen);
    }

    [Fact]
    public void Downgrade_NeedsConfirmation_BeforeItIsOffered()
    {
        using var services = new TestServices();
        services.Updates.Channel = Preview;
        services.Updates.CurrentVersion = "0.2.0-alpha.5";
        services.Updates.ResultsByChannel[Stable] = UpdateCheckResult.Found(new UpdateOffer("0.1.2", null, 1, IsDowngrade: true));
        using var updates = Create(services);

        updates.SelectedChannel = UpdateChannelChoice.For(UpdateChannelKind.Stable);

        Assert.True(updates.HasDowngradeOffer);
        Assert.False(updates.HasOffer);
        Assert.False(updates.IsDialogOpen);
        Assert.False(updates.DownloadCommand.CanExecute(null));
        Assert.Contains("downgrade", updates.DowngradeText, StringComparison.OrdinalIgnoreCase);

        updates.ConfirmDowngradeCommand.Execute(null);

        Assert.False(updates.HasDowngradeOffer);
        Assert.Equal("0.1.2", updates.Offer?.Version);
        Assert.True(updates.DownloadCommand.CanExecute(null));
    }

    [Fact]
    public void StayingOnTheCurrentVersion_IsRemembered_UntilTheChannelCatchesUp()
    {
        using var services = new TestServices();
        services.Updates.Channel = Preview;
        services.Updates.CurrentVersion = "0.2.0-alpha.5";
        services.Updates.ResultsByChannel[Stable] = UpdateCheckResult.Found(new UpdateOffer("0.1.2", null, 1, IsDowngrade: true));
        using var updates = Create(services);
        updates.SelectedChannel = UpdateChannelChoice.For(UpdateChannelKind.Stable);

        updates.StayOnCurrentCommand.Execute(null);
        _ = updates.CheckAsync(manual: true);

        Assert.False(updates.HasDowngradeOffer);
        Assert.False(updates.HasOffer);
        Assert.Contains("until the Stable channel catches up", updates.StatusText);
        Assert.Equal("stable", services.Settings.Current.UpdateChannel);

        services.Updates.ResultsByChannel[Stable] = UpdateCheckResult.Found(new UpdateOffer("0.2.0", null, 1));
        _ = updates.CheckAsync(manual: true);

        Assert.Equal("0.2.0", updates.Offer?.Version);
    }

    [Fact]
    public void Unsupported_BuildsCannotSwitch()
    {
        using var services = new TestServices();
        services.Updates.Support = UpdateSupport.Unsupported("portable");
        services.Updates.Channel = null;
        using var updates = Create(services);

        Assert.False(updates.CanSwitchChannel);
        updates.SelectedChannel = UpdateChannelChoice.For(UpdateChannelKind.Preview);

        Assert.Equal(0, services.Updates.Checks);
        Assert.Equal("unstable", services.Settings.Current.UpdateChannel);
    }
}
