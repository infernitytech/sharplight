using Sharplight.Hub;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Settings;

namespace Sharplight.Studio.Tests;

public sealed class StoreLocationTests
{
    [Fact]
    public void NoChoice_MeansTheDefault()
    {
        Assert.Null(StoreLocation.ToStoreSettings(new AppSettings()));
        Assert.Equal(ModelStoreLocation.Application, StoreLocation.Effective(new AppSettings()));
    }

    [Fact]
    public void ADirectoryFromEarlierVersions_IsACustomFolder()
    {
        var settings = new AppSettings { ModelsDirectory = " /models " };
        Assert.Equal(new ModelStoreSettings { Location = ModelStoreLocation.Custom, CustomPath = "/models" }, StoreLocation.ToStoreSettings(settings));
        Assert.Equal(ModelStoreLocation.Custom, StoreLocation.Effective(settings));
    }

    [Fact]
    public void TheChosenLocation_Wins_AndKeepsTheCustomPath()
    {
        var settings = new AppSettings { ModelsLocation = ModelStoreLocation.User, ModelsDirectory = "/models" };
        Assert.Equal(new ModelStoreSettings { Location = ModelStoreLocation.User, CustomPath = "/models" }, StoreLocation.ToStoreSettings(settings));
        Assert.Equal(ModelStoreLocation.Application, StoreLocation.ToStoreSettings(new AppSettings { ModelsLocation = ModelStoreLocation.Application })!.Location);
    }

    [Fact]
    public void CommandLine_WinsOverTheSettings()
    {
        var resolution = StoreLocation.Resolve("/cli", new AppSettings { ModelsLocation = ModelStoreLocation.User });
        Assert.Equal(("/cli", ModelStoreSource.CommandLine), (resolution.Root, resolution.Source));
    }

    [Fact]
    public void SettingsService_ResolvesTheChosenFolder_AndFollowsChanges()
    {
        using var dir = new TempDir();
        var settings = new JsonSettingsService(dir.File("settings.json"), null);
        settings.Update(current => current with { ModelsLocation = ModelStoreLocation.Custom, ModelsDirectory = dir.File("custom") });
        if (Environment.GetEnvironmentVariable(ModelStore.EnvironmentVariable) is { Length: > 0 })
        {
            return;   // SHARPLIGHT_MODELS overrides the settings in this environment
        }

        Assert.Equal(dir.File("custom"), settings.StoreRoot);
        Assert.Equal("Settings: custom folder", settings.StoreRootSource);
        settings.Update(current => current with { ModelsLocation = ModelStoreLocation.User });
        Assert.Equal(ModelStore.UserRoot, settings.StoreRoot);
    }
}
