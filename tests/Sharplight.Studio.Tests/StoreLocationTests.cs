using Sharplight.Hub;
using Sharplight.Studio.Services.Settings;

namespace Sharplight.Studio.Tests;

public sealed class StoreLocationTests
{
    [Fact]
    public void CommandLine_WinsOverSettings_EnvironmentAndDefaults()
    {
        Assert.Equal(("/cli", "--models-dir"), StoreLocation.Resolve("/cli", "/settings", "/env", "/app", "/user"));
        Assert.Equal(("/settings", "Settings"), StoreLocation.Resolve(null, "/settings", "/env", "/app", "/user"));
        Assert.Equal(("/env", ModelStore.EnvironmentVariable), StoreLocation.Resolve(" ", "", "/env", "/app", "/user"));
    }

    [Fact]
    public void PortableFolder_NextToTheApp_BeatsTheUserDefault()
    {
        using var dir = new TempDir();
        Assert.Equal("/user", StoreLocation.Resolve(null, null, null, dir.Path, "/user").Root);
        Directory.CreateDirectory(dir.File("models"));
        Assert.Equal(dir.File("models"), StoreLocation.Resolve(null, null, null, dir.Path, "/user").Root);
    }
}
