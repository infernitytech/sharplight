using Sharplight.Cpu;
using Sharplight.Runtime;
using Xunit;

namespace Sharplight.Package.Tests;

/// <summary>
/// <see cref="PackagePrecompiler"/> over the two-stage provider contract: portable code then device code per device, sharing between
/// identical devices, empty stages of the CPU, skipped providers, the in-place write (round trip, idempotence, crash safety).
/// </summary>
public sealed class PackagePrecompilerTests
{
    private static readonly Lazy<IComputeDevice> s_cpu = new(() =>
    {
        var provider = new CpuExecutionProvider();
        return provider.OpenDevice(provider.GetDevices()[0]);
    });

    private static PrecompileOptions Options => new() { SymbolBounds = new Dictionary<string, long> { ["T"] = 4 } };

    [Fact]
    public async Task IdenticalDevicesShareTheirCodeAndEveryStageIsStored()
    {
        using var tmp = new TempDir();
        var path = Package(tmp);
        // Two identical GPUs (same driver) and one on another driver.
        var provider = new FakeCodeProvider(s_cpu.Value, ["1", "1", "2"]);
        var progress = new List<PrecompileProgress>();
        var report = await PackagePrecompiler.PrecompileAsync(path, [provider], Options, new SyncProgress(progress.Add), TestContext.Current.CancellationToken);

        Assert.Equal(1, provider.PortableCalls);
        Assert.Equal(2, provider.DeviceCalls);
        Assert.Equal(3, report.Devices.Count);
        Assert.Null(report.Devices[0].Portable!.SharedWith);
        Assert.Equal("Recording GPU 0", report.Devices[1].Portable!.SharedWith);
        Assert.Equal("Recording GPU 0", report.Devices[1].DeviceCode!.SharedWith);
        Assert.Null(report.Devices[2].DeviceCode!.SharedWith);
        Assert.Equal(3, report.StoredSets.Count);
        Assert.Contains(progress, p => p.Stage == PrecompileStage.Portable && p.Kernels is not null);
        Assert.Contains(progress, p => p.Stage == PrecompileStage.DeviceSpecific);
        Assert.Contains(progress, p => p.Stage == PrecompileStage.Writing);

        using var file = new PackageFile(path);
        Assert.Equal(1, file.ArtifactSets.Count(s => s.Tier == ArtifactTiers.Code));
        Assert.Equal(2, file.ArtifactSets.Count(s => s.Tier == ArtifactTiers.Binary));
        // A load on a device with driver 2 sees its own device code and the shared portable code.
        var device = new RecordingArtifactDevice(s_cpu.Value, "2");
        var cache = new CompilationArtifacts(file).For(device)!;
        Assert.True(cache.TryGet("kernel/encode/0", out _));
        Assert.True(cache.TryGet("fake/binary/kernel/encode/0", out _));
        Assert.Single(CompilationArtifacts.Matching(file, device, device.ArtifactTargets[1]));
    }

    [Fact]
    public async Task PrecompilingAgainStoresNothingNew()
    {
        using var tmp = new TempDir();
        var path = Package(tmp);
        var provider = new FakeCodeProvider(s_cpu.Value, ["1"]);
        await PackagePrecompiler.PrecompileAsync(path, [provider], Options, cancellationToken: TestContext.Current.CancellationToken);
        long length = new FileInfo(path).Length;

        var again = await PackagePrecompiler.PrecompileAsync(path, [provider], Options, cancellationToken: TestContext.Current.CancellationToken);
        Assert.Empty(again.StoredSets);
        Assert.Equal(2, again.UnchangedSets.Count);
        Assert.Equal(length, new FileInfo(path).Length);
    }

    [Fact]
    public async Task CpuStagesAreEmptyAndUnavailableOrFailingProvidersAreSkipped()
    {
        using var tmp = new TempDir();
        var path = Package(tmp);
        var cpu = new CpuExecutionProvider();
        Assert.Null(cpu.PortableCodeFormat);
        var request = new PortableCompilationRequest(null, (_, _, _) => throw new InvalidOperationException("the CPU compiles nothing"));
        Assert.True((await cpu.CompilePortableAsync(request, TestContext.Current.CancellationToken)).IsEmpty);
        Assert.True((await cpu.CompileDeviceSpecificAsync(s_cpu.Value, PortableCodeSet.Empty("cpu"), TestContext.Current.CancellationToken)).IsEmpty);

        var failing = new FakeCodeProvider(s_cpu.Value, ["1", "9"]) { FailingDriver = "9" };
        long before = new FileInfo(path).Length;
        var report = await PackagePrecompiler.PrecompileAsync(path, [cpu, new UnavailableFake(), failing], Options,
            cancellationToken: TestContext.Current.CancellationToken);

        var cpuResult = report.Devices.Single(d => d.Provider == "cpu");
        Assert.Null(cpuResult.Portable);
        Assert.Null(cpuResult.DeviceCode);
        Assert.Contains("nothing to precompile", cpuResult.Note);
        Assert.Contains(report.Skipped, s => s.Provider == "missing" && s.Reason.Contains("no driver", StringComparison.Ordinal));
        Assert.Contains(report.Skipped, s => s.Provider == "fake" && s.Device == "Recording GPU 1" && s.Reason.Contains("driver error", StringComparison.Ordinal));
        Assert.Equal(2, report.StoredSets.Count);
        Assert.True(new FileInfo(path).Length > before);
    }

    [Fact]
    public async Task CodeTierOnlySkipsTheDeviceStage()
    {
        using var tmp = new TempDir();
        var path = Package(tmp);
        var provider = new FakeCodeProvider(s_cpu.Value, ["1"]);
        var options = new PrecompileOptions { SymbolBounds = Options.SymbolBounds, Tiers = new HashSet<string> { ArtifactTiers.Code } };
        var report = await PackagePrecompiler.PrecompileAsync(path, [provider], options, cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal(0, provider.DeviceCalls);
        Assert.Single(report.StoredSets);
    }

    [Fact]
    public void AFailedWriteLeavesThePackageByteForByteUnchanged()
    {
        using var tmp = new TempDir();
        var path = Package(tmp);
        ModelPackage.AddArtifacts(path, [Set(FakeArtifactDevice.Code(), "kernel/a")]);
        var original = File.ReadAllBytes(path);

        Assert.Throws<IOException>(() => ModelPackage.AddArtifacts(path, FailingSets()));
        Assert.Equal(original, File.ReadAllBytes(path));
    }

    [Fact]
    public void AnInterruptedAppendLeavesTheOldContentReadable()
    {
        using var tmp = new TempDir();
        var path = Package(tmp);
        ModelPackage.AddArtifacts(path, [Set(FakeArtifactDevice.Code(), "kernel/a")]);
        // A crash after writing sections but before the header switched: bytes after the old table, header unchanged.
        using (var stream = new FileStream(path, FileMode.Append))
        {
            stream.Write(Enumerable.Range(0, 4096).Select(i => (byte)(i * 7)).ToArray());
        }

        using (var file = new PackageFile(path))
        {
            var set = Assert.Single(file.ArtifactSets);
            Assert.True(set.Entries.TryGet("kernel/a", out _));
        }

        // The dead bytes do not stop later writes.
        ModelPackage.AddArtifacts(path, [Set(FakeArtifactDevice.Binary(), "fake/b")]);
        using var after = new PackageFile(path);
        Assert.Equal(2, after.ArtifactSets.Count);
    }

    private static string Package(TempDir tmp)
    {
        var path = tmp.File("model.slm");
        ModelPackage.Save(path, ModelDescriptionPackageTests.SpeechDefinition(), []);
        return path;
    }

    private static CompilationCache Set(IReadOnlyDictionary<string, string> descriptor, string key)
    {
        var set = new CompilationCache(descriptor);
        set.Put(key, new byte[] { 1, 2, 3 });
        return set;
    }

    /// <summary>One set is written, then the write fails (a full disk, a crash of the producer).</summary>
    private static IEnumerable<CompilationCache> FailingSets()
    {
        yield return Set(FakeArtifactDevice.Binary(), "fake/x");
        throw new IOException("disk full");
    }

    private sealed class SyncProgress(Action<PrecompileProgress> handler) : IProgress<PrecompileProgress>
    {
        public void Report(PrecompileProgress value) => handler(value);
    }

    private sealed class UnavailableFake : IExecutionProvider
    {
        public string Name => "missing";

        public string? PortableCodeFormat => "none";

        public ProviderAvailability CheckAvailability() => ProviderAvailability.Unavailable("no driver");

        public IReadOnlyList<DeviceInfo> GetDevices() => [];

        public IComputeDevice OpenDevice(DeviceInfo device, DeviceOptions? options = null) => throw new NotSupportedException();

        public bool CanCompileFor(DeviceProfile profile) => false;

        public Task<PortableCodeSet> CompilePortableAsync(PortableCompilationRequest request, CancellationToken cancellationToken = default)
            => throw new NotSupportedException();

        public Task<DeviceCodeSet> CompileDeviceSpecificAsync(IComputeDevice device, PortableCodeSet portable, CancellationToken cancellationToken = default)
            => throw new NotSupportedException();

        public void Dispose()
        {
        }
    }
}
