using Sharplight.Studio.Services.Updates;

namespace Sharplight.Studio.Tests;

public sealed class UpdateEndpointTests
{
    [Fact]
    public void Resolve_PrefersEnvironment_ThenSetting_ThenBakedUrl()
    {
        Assert.Equal("https://env.example/", UpdateEndpoint.Resolve("https://env.example", "https://set.example", "https://baked.example"));
        Assert.Equal("https://set.example/studio/", UpdateEndpoint.Resolve(null, " https://set.example/studio ", "https://baked.example"));
        Assert.Equal("https://baked.example/", UpdateEndpoint.Resolve("", null, "https://baked.example/"));
    }

    [Fact]
    public void Resolve_IsNull_WhenNothingUsableIsConfigured()
    {
        Assert.Null(UpdateEndpoint.Resolve(null, null, null));
        Assert.Null(UpdateEndpoint.Resolve("", "  ", "not a url"));
        Assert.Null(UpdateEndpoint.Resolve("ftp://example.test/", null, "file:///tmp/x"));
    }

    [Fact]
    public void Resolve_SkipsAnInvalidHigherSource()
    {
        Assert.Equal("https://baked.example/", UpdateEndpoint.Resolve("nonsense", null, "https://baked.example"));
    }

    [Fact]
    public async Task VelopackService_IsUnsupported_WhenNotInstalled()
    {
        var service = new VelopackUpdateService("https://updates.example.test/");

        Assert.False(service.Support.IsSupported);
        Assert.Contains("unavailable for this build", service.Support.Reason);
        Assert.Null(service.Channel);
        var result = await service.CheckAsync(null, CancellationToken.None);
        Assert.NotNull(result.Error);
    }
}
