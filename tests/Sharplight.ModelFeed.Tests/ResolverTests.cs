using Sharplight.ModelFeed.Resolution;
using Sharplight.ModelFeed.Versioning;

namespace Sharplight.ModelFeed.Tests;

public sealed class ResolverTests
{
    private static readonly Dictionary<string, IReadOnlyList<PackageCandidate>> Feed = new()
    {
        ["whisper.q5"] =
        [
            Candidate("whisper.q5", "1.0.0", Dependency("vad.onnx", "[1.0.0,2.0.0)", role: "voice-activity")),
            Candidate("whisper.q5", "1.1.0", Dependency("vad.onnx", "[1.1.0,2.0.0)", role: "voice-activity")),
            Candidate("whisper.q5", "2.0.0-rc.1"),
        ],
        ["vad.onnx"] = [Candidate("vad.onnx", "1.0.0"), Candidate("vad.onnx", "1.1.0"), Candidate("vad.onnx", "1.2.0"), Candidate("vad.onnx", "2.0.0")],
        ["embed.f32"] = [Candidate("embed.f32", "1.0.0", Dependency("rerank.f32", null, DependencyKinds.Optional, "pairing"))],
        ["rerank.f32"] = [Candidate("rerank.f32", "1.0.0")],
        ["needs-new-server.q8"] = [Candidate("needs-new-server.q8", "1.0.0") with { Requirements = new PackageRequirements { Server = ">=2.0.0" } }],
        ["broken.q8"] = [Candidate("broken.q8", "1.0.0", Dependency("missing.pkg", "[1.0.0,)"))],
        ["conflict.a"] = [Candidate("conflict.a", "1.0.0", Dependency("vad.onnx", "[2.0.0,3.0.0)"), Dependency("whisper.q5", "[1.0.0]"))],
        ["yanked.q8"] =
        [
            Candidate("yanked.q8", "1.0.0"),
            Candidate("yanked.q8", "1.1.0") with { State = new PackageState { Yanked = new PackageYank { Reason = "broken" } } },
        ],
    };

    [Fact]
    public async Task ChoosesHighestVersions_AndRequiredDependencies()
    {
        var result = await ResolveAsync([new PackageRequest("whisper.q5")]);
        Assert.True(result.Succeeded);
        Assert.Equal(["whisper.q5@1.1.0", "vad.onnx@1.2.0"], result.Packages.Select(package => $"{package.Id}@{package.Version}"));
        var vad = result.Packages[1];
        Assert.Equal(["whisper.q5"], vad.RequiredBy);
        Assert.Equal("voice-activity", vad.Role);
    }

    [Fact]
    public async Task KeepsPreferredVersions_WhenTheySatisfy()
    {
        var options = new ResolutionOptions { Preferred = new Dictionary<string, string> { ["vad.onnx"] = "1.1.0", ["whisper.q5"] = "1.0.0" } };
        var result = await ResolveAsync([new PackageRequest("whisper.q5", "[1.1.0,)")], options);
        Assert.Equal(["whisper.q5@1.1.0", "vad.onnx@1.1.0"], result.Packages.Select(package => $"{package.Id}@{package.Version}"));
    }

    [Fact]
    public async Task PinnedVersion_IsHonoured()
    {
        var result = await ResolveAsync([new PackageRequest("whisper.q5", "[1.0.0]")]);
        Assert.Equal("1.0.0", result.Packages[0].Version);
    }

    [Fact]
    public async Task Prereleases_OnlyWhenAllowed()
    {
        Assert.Equal("1.1.0", (await ResolveAsync([new PackageRequest("whisper.q5")])).Packages[0].Version);
        Assert.Equal("2.0.0-rc.1", (await ResolveAsync([new PackageRequest("whisper.q5")], new ResolutionOptions { IncludePrerelease = true })).Packages[0].Version);
    }

    [Fact]
    public async Task OptionalDependencies_OnlyWhenSelected()
    {
        Assert.Single((await ResolveAsync([new PackageRequest("embed.f32")])).Packages);
        var withPairing = await ResolveAsync([new PackageRequest("embed.f32")], new ResolutionOptions { IncludeOptional = new HashSet<string> { "rerank.f32" } });
        Assert.Equal(2, withPairing.Packages.Count);
    }

    [Fact]
    public async Task IncompatibleServer_FailsWithTheReason()
    {
        var options = new ResolutionOptions { Environment = new CompatibilityEnvironment { HostVersion = SemanticVersion.Parse("1.0.0") } };
        var result = await ResolveAsync([new PackageRequest("needs-new-server.q8")], options);
        Assert.False(result.Succeeded);
        Assert.Contains("needs server >=2.0.0; this is 1.0.0", result.Problems[0], StringComparison.Ordinal);
        Assert.Empty(result.Packages);
    }

    [Fact]
    public async Task MissingDependency_AndConflicts_FailWithoutPartialResults()
    {
        var missing = await ResolveAsync([new PackageRequest("broken.q8")]);
        Assert.False(missing.Succeeded);
        Assert.Contains("missing.pkg", missing.Problems[0], StringComparison.Ordinal);

        var conflict = await ResolveAsync([new PackageRequest("conflict.a")]);
        Assert.False(conflict.Succeeded);
        Assert.Contains("vad.onnx", string.Join(" ", conflict.Problems), StringComparison.Ordinal);
    }

    [Fact]
    public async Task YankedVersions_AreNeverChosen()
    {
        Assert.Equal("1.0.0", (await ResolveAsync([new PackageRequest("yanked.q8")])).Packages[0].Version);
        Assert.False((await ResolveAsync([new PackageRequest("yanked.q8", "[1.1.0]")])).Succeeded);
    }

    [Fact]
    public void Compatibility_ChecksEngineAndFormat()
    {
        var environment = new CompatibilityEnvironment { EngineVersion = SemanticVersion.Parse("0.3.0"), MaxPackageFormat = 3 };
        Assert.Empty(PackageCompatibility.Check(new PackageRequirements { Engine = ">=0.3.0", PackageFormat = 3 }, environment));
        Assert.Equal(2, PackageCompatibility.Check(new PackageRequirements { Engine = ">=0.4.0", PackageFormat = 4 }, environment).Count);
    }

    private static Task<ResolutionResult> ResolveAsync(IReadOnlyList<PackageRequest> requests, ResolutionOptions? options = null) =>
        new DependencyResolver().ResolveAsync(requests, (id, _) => Task.FromResult(Feed.GetValueOrDefault(id) ?? []), options, TestContext.Current.CancellationToken);

    private static PackageCandidate Candidate(string id, string version, params PackageDependency[] dependencies) =>
        new() { Id = id, Version = version, Dependencies = dependencies };

    private static PackageDependency Dependency(string id, string? range, string kind = DependencyKinds.Required, string? role = null) =>
        new() { Id = id, Range = range, Kind = kind, Role = role };
}
