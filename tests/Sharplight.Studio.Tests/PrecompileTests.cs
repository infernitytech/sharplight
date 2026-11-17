using Sharplight.Cpu;
using Sharplight.Package;
using Sharplight.Runtime;
using Sharplight.Studio.Services.Library;
using Sharplight.Studio.Tests.Fakes;
using Sharplight.Studio.ViewModels.Inspector;
using Sharplight.Studio.ViewModels.Models;

namespace Sharplight.Studio.Tests;

/// <summary>
/// The library's Precompile action: progress per provider and device, the result summary, refusal while the model is resident, the
/// entry and an open inspector read again afterwards, and the inspector's stages per provider.
/// </summary>
public sealed class PrecompileTests
{
    private static readonly CodeSetSummary s_portable = new("provider=vulkan tier=code kind=spirv", "spirv", 120, 3_000_000, 0, null, TimeSpan.FromSeconds(9));
    private static readonly CodeSetSummary s_binaries = new("provider=vulkan tier=binary kind=pipeline-binary", "pipeline-binary", 120, 8_000_000, 0, null,
        TimeSpan.FromSeconds(4));

    [Fact]
    public async Task PrecompileShowsEveryDeviceAndReadsTheEntryAgain()
    {
        using var tmp = new TempDir();
        var path = tmp.File("model.slm");
        TestPackages.WriteSmall(path);
        using var services = new TestServices();
        services.Library.Entries.Add(new LibraryEntry { Id = "small", DisplayName = "small.slm", Path = path, Format = "SLM" });
        services.Library.Reload = entry => entry with { Precompiled = ["provider=vulkan tier=code", "provider=vulkan tier=binary"] };
        services.Precompiler.Steps.Add(new PrecompileProgress("vulkan", "GPU A", PrecompileStage.Portable, "Compiling",
            new LoadProgress(LoadStage.Compiling, 30, 120, "GPU A")));
        var report = new PackagePrecompileReport(path,
            [
                new DevicePrecompileResult("vulkan", "GPU A", s_portable, s_binaries, null),
                new DevicePrecompileResult("vulkan", "GPU B", s_portable with { SharedWith = "GPU A", Elapsed = TimeSpan.Zero },
                    s_binaries with { SharedWith = "GPU A", Elapsed = TimeSpan.Zero }, null),
                new DevicePrecompileResult("cpu", "Host CPU", null, null, "nothing to precompile (kernels are built into the provider)"),
            ],
            [new SkippedProvider("cuda", "", "not available")], ["a", "b"], [], 11_000_000, 50_000_000, TimeSpan.FromSeconds(13));
        services.Precompiler.Report = report;
        services.Precompiler.Gate = new TaskCompletionSource();
        var library = new LibraryViewModel(services.Services, new FakeNavigator());
        await library.RefreshAsync();
        var entry = library.Entries.Single();
        Assert.True(entry.CanPrecompile);
        Assert.False(entry.IsPrecompiled);

        var running = library.PrecompileAsync(entry);
        Assert.True(entry.IsPrecompiling);
        Assert.False(entry.CanPrecompile);

        // The service runs on a worker (never on the UI thread): wait until it started and reported. The row is added before its
        // progress is applied, so wait for the progress too (the check after the row alone flaked on a loaded machine).
        for (int i = 0; i < 1000 && (!services.Precompiler.IsBusy(path) || entry.PrecompileRows.Count == 0 || entry.PrecompileRows[0].Progress == 0); i++)
        {
            await Task.Delay(10, TestContext.Current.CancellationToken);
        }

        Assert.True(services.Precompiler.IsBusy(path));
        var row = Assert.Single(entry.PrecompileRows);
        Assert.Equal("vulkan · GPU A", row.Title);
        Assert.Equal(0.25, row.Progress, 3);
        Assert.Contains("30 / 120", row.Status, StringComparison.Ordinal);

        services.Precompiler.Gate.SetResult();
        await running;
        Assert.False(entry.IsPrecompiling);
        Assert.Equal(4, entry.PrecompileRows.Count);
        Assert.Contains("shared with GPU A", entry.PrecompileRows.Single(r => r.Device == "GPU B").Status, StringComparison.Ordinal);
        Assert.True(entry.PrecompileRows.Single(r => r.Provider == "cpu").IsSkipped);
        Assert.True(entry.PrecompileRows.Single(r => r.Provider == "cuda").IsSkipped);
        Assert.Contains("Stored 2 artifact sets", entry.PrecompileStatus, StringComparison.Ordinal);
        Assert.Contains("for 2 devices", entry.PrecompileStatus, StringComparison.Ordinal);
        Assert.Contains("cuda (not available)", entry.PrecompileStatus, StringComparison.Ordinal);
        Assert.True(entry.IsPrecompiled);
    }

    [Fact]
    public async Task PrecompileIsRefusedWhileTheModelIsResident()
    {
        using var tmp = new TempDir();
        var path = tmp.File("model.slm");
        TestPackages.WriteSmall(path);
        using var services = new TestServices();
        services.Library.Entries.Add(new LibraryEntry { Id = "small", DisplayName = "small.slm", Path = path, Format = "SLM" });
        services.Runtime.Add(new FakeModel("small") { Path = path });
        var library = new LibraryViewModel(services.Services, new FakeNavigator());
        await library.RefreshAsync();
        var entry = library.Entries.Single();

        Assert.True(entry.IsLoaded);
        Assert.False(entry.CanPrecompile);
        Assert.Contains("Unload the model", entry.PrecompileHint, StringComparison.Ordinal);
        await library.PrecompileAsync(entry);
        Assert.Empty(services.Precompiler.Calls);
        Assert.Contains("Unload small", library.Status, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ModelsThatAreNotPackagesCannotBePrecompiled()
    {
        using var tmp = new TempDir();
        var path = tmp.File("model.gguf");
        File.WriteAllBytes(path, [1, 2, 3]);
        using var services = new TestServices();
        services.Library.Entries.Add(new LibraryEntry { Id = "gguf", DisplayName = "model.gguf", Path = path, Format = "GGUF" });
        var library = new LibraryViewModel(services.Services, new FakeNavigator());
        await library.RefreshAsync();
        var entry = library.Entries.Single();
        Assert.False(entry.CanPrecompile);
        await library.PrecompileAsync(entry);
        Assert.Empty(services.Precompiler.Calls);
    }

    [Fact]
    public async Task AFailedPrecompileSaysThePackageWasNotChanged()
    {
        using var tmp = new TempDir();
        var path = tmp.File("model.slm");
        TestPackages.WriteSmall(path);
        using var services = new TestServices();
        services.Library.Entries.Add(new LibraryEntry { Id = "small", DisplayName = "small.slm", Path = path, Format = "SLM" });
        services.Precompiler.Run = (_, _, _) => throw new IOException("disk full");
        var library = new LibraryViewModel(services.Services, new FakeNavigator());
        await library.RefreshAsync();
        var entry = library.Entries.Single();
        await library.PrecompileAsync(entry);
        Assert.False(entry.IsPrecompiling);
        Assert.Contains("disk full", entry.PrecompileStatus, StringComparison.Ordinal);
        Assert.Contains("not changed", entry.PrecompileStatus, StringComparison.Ordinal);
    }

    [Fact]
    public async Task TheRealPrecompilerRunsForTheCpuAndRefreshesTheOpenInspector()
    {
        using var tmp = new TempDir();
        var path = tmp.File("model.slm");
        TestPackages.WriteDescribed(path);
        using var services = new TestServices();
        services.Library.Entries.Add(new LibraryEntry { Id = "described", DisplayName = "described.slm", Path = path, Format = "SLM" });
        services.Precompiler.Run = (package, progress, cancellationToken) =>
            PackagePrecompiler.PrecompileAsync(package, [new CpuExecutionProvider()], new PrecompileOptions(), progress, cancellationToken);
        var library = new LibraryViewModel(services.Services, new FakeNavigator());
        await library.RefreshAsync();
        var entry = library.Entries.Single();
        Assert.True(await library.OpenInspectorAsync(path, null));
        var inspector = library.Inspector!;
        var before = inspector.Package;
        long length = new FileInfo(path).Length;

        await library.PrecompileAsync(entry);
        Assert.Contains("Already up to date for 0 devices", entry.PrecompileStatus, StringComparison.Ordinal);
        Assert.Contains("nothing to precompile", Assert.Single(entry.PrecompileRows).Status, StringComparison.Ordinal);
        Assert.Equal(length, new FileInfo(path).Length);
        Assert.NotSame(before, inspector.Package);
    }

    [Fact]
    public void InspectorShowsTheStagesPerProviderAndTarget()
    {
        using var tmp = new TempDir();
        var path = tmp.File("model.slm");
        TestPackages.WriteSmall(path);
        var code = Set(new Dictionary<string, string>
        {
            [ArtifactDescriptorKeys.Provider] = "zeta",
            [ArtifactDescriptorKeys.Tier] = ArtifactTiers.Code,
            [ArtifactDescriptorKeys.Kind] = "spirv",
            [ArtifactDescriptorKeys.Target] = "spirv_1_6",
            [ArtifactDescriptorKeys.Subgroup] = "32",
        }, "kernel/a");
        var binary = Set(new Dictionary<string, string>
        {
            [ArtifactDescriptorKeys.Provider] = "zeta",
            [ArtifactDescriptorKeys.Tier] = ArtifactTiers.Binary,
            [ArtifactDescriptorKeys.Kind] = "pipeline-binary",
            [ArtifactDescriptorKeys.DeviceName] = "GPU A",
            [ArtifactDescriptorKeys.Driver] = "1f",
        }, "vulkan/pipeline_binary/a");
        ModelPackage.AddArtifacts(path, [binary, code]);

        var artifacts = new ArtifactsViewModel(ModelPackageReader.Open(path).ArtifactSets);
        // Sorted by provider (the test package's own sets come first), then portable before device code.
        var stages = artifacts.Stages.Where(s => s.Provider == "zeta").ToList();
        Assert.Equal(["Portable code", "Device code"], stages.Select(s => s.Stage));
        Assert.Contains("GPU A", stages[1].Target, StringComparison.Ordinal);
        Assert.Equal("zeta · Device code", artifacts.Sets[^1].Heading);
    }

    private static CompilationCache Set(Dictionary<string, string> descriptor, string key)
    {
        var set = new CompilationCache(descriptor);
        set.Put(key, new byte[] { 1, 2, 3, 4 });
        return set;
    }
}
