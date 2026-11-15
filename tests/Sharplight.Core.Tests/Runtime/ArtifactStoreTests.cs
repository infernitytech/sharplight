using Sharplight.Runtime;
using Xunit;

namespace Sharplight.Core.Tests.Runtime;

/// <summary>
/// The per-user artifact store: the artifact sets of a package (same descriptors, ids, keys and matching) in a directory shared by
/// processes, with validated entries and descriptors, consulted after the package's own sets.
/// </summary>
public sealed class ArtifactStoreTests : IDisposable
{
    private readonly string _directory = Directory.CreateTempSubdirectory("sharplight-store-test-").FullName;

    public void Dispose() => Directory.Delete(_directory, recursive: true);

    [Fact]
    public void SetsRoundTripAcrossProcessesWithThePackageDescriptors()
    {
        var target = CodeTarget("spirv_1_6");
        new ArtifactStore(_directory).SetFor(target).Entries.Put("kernel/a", new byte[] { 1, 2, 3 });

        // A second instance reads the directory like another process would.
        var other = new ArtifactStore(_directory);
        var set = Assert.Single(other.ArtifactSets);
        Assert.Equal(target.Descriptor, set.Descriptor);
        Assert.Equal(target.Id, ArtifactDescriptor.Id(set.Descriptor));
        Assert.True(set.Entries.TryGet("kernel/a", out var data));
        Assert.Equal(new byte[] { 1, 2, 3 }, data.ToArray());
        Assert.False(set.Entries.TryGet("kernel/b", out _));

        // Matching is the package's: a target of another profile does not consult the set.
        Assert.True(target.Matches(set.Descriptor));
        Assert.False(CodeTarget("spirv_1_5").Matches(set.Descriptor));
    }

    [Fact]
    public void DeviceCodeIsStoredAsItsOwnSet()
    {
        var store = new ArtifactStore(_directory);
        var descriptor = new Dictionary<string, string>
        {
            [ArtifactDescriptorKeys.Provider] = "test",
            [ArtifactDescriptorKeys.Tier] = ArtifactTiers.Binary,
            [ArtifactDescriptorKeys.Kind] = "binary",
            [ArtifactDescriptorKeys.Driver] = "1",
        };
        store.Add(new DeviceCodeSet("test", "Test GPU", descriptor, new Dictionary<string, ReadOnlyMemory<byte>> { ["bin/a"] = new byte[] { 7 } }));
        store.Add(DeviceCodeSet.Empty("cpu", "CPU"));

        var set = Assert.Single(new ArtifactStore(_directory).ArtifactSets);
        Assert.Equal(ArtifactTiers.Binary, set.Tier);
        Assert.True(set.Entries.TryGet("bin/a", out var data));
        Assert.Equal(new byte[] { 7 }, data.ToArray());
    }

    [Fact]
    public void DamagedDescriptorsAndEntriesAreIgnored()
    {
        var target = CodeTarget("spirv_1_6");
        new ArtifactStore(_directory).SetFor(target).Entries.Put("kernel/a", new byte[] { 1, 2, 3 });
        var setDirectory = Path.Combine(_directory, target.Id);

        // An entry file that no longer validates is a miss (and is removed).
        var entry = Directory.GetFiles(setDirectory).Single(f => !f.EndsWith(".json", StringComparison.Ordinal));
        var bytes = File.ReadAllBytes(entry);
        bytes[^1] ^= 0xff;
        File.WriteAllBytes(entry, bytes);
        Assert.False(new ArtifactStore(_directory).ArtifactSets.Single().Entries.TryGet("kernel/a", out _));
        Assert.False(File.Exists(entry));

        // A descriptor that does not hash to its directory (edited, or foreign) or does not parse: the set is not consulted.
        File.WriteAllText(Path.Combine(setDirectory, "descriptor.json"), """{ "provider": "test", "tier": "code", "kind": "other" }""");
        Assert.Empty(new ArtifactStore(_directory).ArtifactSets);
        File.WriteAllText(Path.Combine(setDirectory, "descriptor.json"), "{ trunc");
        Assert.Empty(new ArtifactStore(_directory).ArtifactSets);
    }

    [Fact]
    public void ModelSetsComeFirstThenTheStoreAndCompilationsAreWrittenThrough()
    {
        var target = CodeTarget("spirv_1_6");
        var store = new ArtifactStore(_directory).SetFor(target).Entries;
        store.Put("kernel/stored", new byte[] { 2 });
        var package = new CompilationCache(target.Descriptor);
        package.Put("kernel/stored", new byte[] { 1 });

        var cache = new CompilationCache(target.Descriptor, [package], store);
        Assert.True(cache.TryGet("kernel/stored", out var first));
        Assert.Equal(new byte[] { 1 }, first.ToArray());
        Assert.Equal(1, cache.FallbackHits);

        // Served from the store: recorded as new for the model (it is not in the package yet).
        store.Put("kernel/only-in-store", new byte[] { 3 });
        Assert.True(cache.TryGet("kernel/only-in-store", out var second));
        Assert.Equal(new byte[] { 3 }, second.ToArray());
        Assert.Equal(2, cache.Entries.Count);
        Assert.Equal(1, cache.FallbackHits);

        // Compiled: recorded and written through to the store for the next process.
        cache.Put("kernel/compiled", new byte[] { 4 });
        Assert.True(new ArtifactStore(_directory).SetFor(target).Entries.TryGet("kernel/compiled", out var third));
        Assert.Equal(new byte[] { 4 }, third.ToArray());
    }

    [Fact]
    public void DeviceOptionsChooseOrDisableTheStore()
    {
        Assert.Null(ArtifactStore.For(new DeviceOptions { UseCompilationCache = false, KernelCacheDirectory = _directory }));
        var store = ArtifactStore.For(new DeviceOptions { KernelCacheDirectory = _directory });
        Assert.NotNull(store);
        Assert.Equal(Path.GetFullPath(_directory), store.Directory);
        Assert.Same(store, ArtifactStore.For(new DeviceOptions { KernelCacheDirectory = _directory }));
    }

    [Fact]
    public void TheSizeCapCoversEverySetButKeepsDescriptors()
    {
        var store = new ArtifactStore(_directory, maxBytes: 20_000);
        var payload = new byte[1_000];
        for (int i = 0; i < 40; i++)
        {
            store.SetFor(CodeTarget($"profile_{i % 4}")).Entries.Put($"kernel/{i}", payload);
        }

        long entryBytes = Directory.EnumerateFiles(_directory, "*", SearchOption.AllDirectories)
            .Where(f => !f.EndsWith(".json", StringComparison.Ordinal)).Sum(f => new FileInfo(f).Length);
        Assert.InRange(entryBytes, 1, 20_000);
        Assert.Equal(4, new ArtifactStore(_directory).ArtifactSets.Count);
    }

    private static ArtifactTarget CodeTarget(string profile) => new(
        new Dictionary<string, string>
        {
            [ArtifactDescriptorKeys.Provider] = "test",
            [ArtifactDescriptorKeys.Tier] = ArtifactTiers.Code,
            [ArtifactDescriptorKeys.Kind] = "code",
            [ArtifactDescriptorKeys.Target] = profile,
        },
        [ArtifactDescriptorKeys.Target],
        ["kernel/"]);
}
