using Sharplight.Studio.Services;

namespace Sharplight.Studio.Tests;

public sealed class ResolveOnDemandTests
{
    [Fact]
    public void LocalFiles_ResolveByFileName_OnlyWhenOnDemandLoadingIsOn()
    {
        using var services = new TestServices();
        var file = Path.Combine(services.Directory, "Tiny-Model.slm");
        File.WriteAllBytes(file, [0]);
        services.Settings.Update(current => current with { LocalModels = [file], DefaultLoad = current.DefaultLoad with { ContextLength = 4096 } });

        Assert.Null(StudioHost.ResolveOnDemand(services.Settings, "tiny-model"));

        services.Settings.Update(current => current with { Residency = current.Residency with { LoadOnDemand = true } });
        var request = StudioHost.ResolveOnDemand(services.Settings, "tiny-model");
        Assert.NotNull(request);
        Assert.Equal(("tiny-model", file, 4096), (request.Id, request.Path, request.Settings.ContextLength));
        Assert.Null(StudioHost.ResolveOnDemand(services.Settings, "unknown"));
    }
}
