using System.Buffers.Binary;
using System.Text;
using Sharplight.Models;
using Sharplight.Runtime;
using Xunit;

namespace Sharplight.Package.Tests;

public class ArtifactSetTests
{
    private static CompilationCache Set(IReadOnlyDictionary<string, string> descriptor, params (string Key, byte[] Data)[] entries)
    {
        var set = new CompilationCache(descriptor);
        foreach (var (key, data) in entries)
        {
            set.Put(key, data);
        }

        return set;
    }

    private static byte[] Get(ICompilationCache cache, string key)
    {
        Assert.True(cache.TryGet(key, out var data), $"{key} not found");
        return data.ToArray();
    }

    private static void Dispose(ModelSource source)
    {
        foreach (var r in source.OwnedResources)
        {
            r.Dispose();
        }
    }

    [Fact]
    public void ArtifactSetsAreStoredReplacedAndAppended()
    {
        using var tmp = new TempDir();
        var p = tmp.File("a.slm");
        var code = Set(FakeArtifactDevice.Code(), ("kernel/1", new byte[] { 1, 2, 3 }));
        var binary = Set(FakeArtifactDevice.Binary(), (FakeArtifactDevice.BinaryKey, new byte[100]));
        ModelPackage.Save(p, PackageTests.SmallDefinition(), [code, binary, new CompilationCache(FakeArtifactDevice.Code("other"))]);

        using (var f = new PackageFile(p))
        {
            Assert.Equal(2, f.ArtifactSets.Count);
            var storedCode = f.ArtifactSets.Single(s => s.Tier == ArtifactTiers.Code);
            Assert.Equal(FakeArtifactDevice.Code().OrderBy(kv => kv.Key), storedCode.Descriptor.OrderBy(kv => kv.Key));
            Assert.Equal(new byte[] { 1, 2, 3 }, Get(storedCode.Entries, "kernel/1"));
            Assert.False(storedCode.Entries.TryGet("kernel/2", out _));
            Assert.Throws<NotSupportedException>(() => storedCode.Entries.Put("kernel/2", new byte[1]));

            // Append while the file is open: the open mapping keeps serving its data.
            var code2 = Set(FakeArtifactDevice.Code(), ("kernel/1", new byte[] { 9 }));
            var other = Set(FakeArtifactDevice.Code("spirv_1_5"), ("kernel/7", new byte[] { 7, 7 }));
            var ids = ModelPackage.AddArtifacts(p, [code2, other]);
            Assert.Equal([ArtifactDescriptor.Id(code2.Descriptor), ArtifactDescriptor.Id(other.Descriptor)], ids);
            Assert.Equal(new byte[] { 1, 2, 3 }, Get(storedCode.Entries, "kernel/1"));
        }

        using (var f = new PackageFile(p))
        {
            Assert.Equal(3, f.ArtifactSets.Count);
            // Newest first: the replacement and the new set come before the untouched binary set.
            Assert.Equal(["spirv_1_5", "spirv_1_6", ""], f.ArtifactSets.Select(s => ArtifactDescriptor.Get(s.Descriptor, ArtifactDescriptorKeys.Target)));
            Assert.Equal(new byte[] { 9 }, Get(f.ArtifactSets[1].Entries, "kernel/1"));
            Assert.Equal(PackageTests.SmallDefinition().Weights["w"].ReadAll(), f.Weights["w"].ReadAll());
        }

        Assert.Equal(ModelPackage.Version, ModelPackageReader.Open(p).Version);
    }

    [Fact]
    public void TargetsMatchOnTheirSignificantKeysOnly()
    {
        var device = new FakeArtifactDevice();
        var code = device.ArtifactTargets[0];
        var binary = device.ArtifactTargets[1];

        // Code entries are content addressed: another subgroup size still matches, another code target or provider does not.
        Assert.True(code.Matches(FakeArtifactDevice.Code(subgroup: 64)));
        Assert.False(code.Matches(FakeArtifactDevice.Code("spirv_1_5")));
        Assert.False(code.Matches(FakeArtifactDevice.Code(provider: "other")));
        Assert.False(code.Matches(FakeArtifactDevice.Binary()));
        // Binaries need the same device and driver.
        Assert.True(binary.Matches(FakeArtifactDevice.Binary()));
        Assert.False(binary.Matches(FakeArtifactDevice.Binary(driver: "2")));
        var withoutDevice = FakeArtifactDevice.Binary();
        withoutDevice.Remove(ArtifactDescriptorKeys.Device);
        Assert.False(binary.Matches(withoutDevice));

        Assert.Throws<ArgumentException>(() => new ArtifactTarget(new Dictionary<string, string> { [ArtifactDescriptorKeys.Provider] = "x" }, [], []));
        Assert.Equal("provider=fake tier=code kind=spirv subgroup=32 target=spirv_1_6", ArtifactDescriptor.Format(code.Descriptor));
        Assert.Equal(ArtifactDescriptor.Id(FakeArtifactDevice.Code()), code.Id);
        Assert.NotEqual(ArtifactDescriptor.Id(FakeArtifactDevice.Code(subgroup: 64)), code.Id);
        Assert.StartsWith("fake.code.spirv.", code.Id);
    }

    [Fact]
    public void DeviceCacheRoutesKeysByTierAndExportsPerTier()
    {
        var device = new FakeArtifactDevice();
        var artifacts = new CompilationArtifacts();
        var cache = artifacts.For(device)!;
        Assert.Same(cache, artifacts.For(device));
        cache.Put("kernel/a", new byte[] { 1 });
        cache.Put("other/x", new byte[] { 2 }); // unclaimed: goes to the catch-all (binary) tier

        var all = artifacts.Export();
        Assert.Equal(1, device.ExportCalls);
        var codeSet = all.Single(s => s.Descriptor[ArtifactDescriptorKeys.Tier] == ArtifactTiers.Code);
        var binarySet = all.Single(s => s.Descriptor[ArtifactDescriptorKeys.Tier] == ArtifactTiers.Binary);
        Assert.Equal(["kernel/a"], codeSet.Entries.Keys);
        Assert.Equal([FakeArtifactDevice.BinaryKey, "other/x"], binarySet.Entries.Keys.Order(StringComparer.Ordinal));

        var codeOnly = artifacts.Export(new HashSet<string> { ArtifactTiers.Code });
        Assert.Same(codeSet, Assert.Single(codeOnly));

        // A device without targets (the CPU) gets no cache.
        Assert.Null(artifacts.For(new Cpu.CpuExecutionProvider().OpenDevice(new Cpu.CpuExecutionProvider().GetDevices()[0])));
    }

    [Fact]
    public void RecordingCacheCapturesFallbackHitsAndNewEntries()
    {
        var stored = Set(FakeArtifactDevice.Code(), ("old", new byte[] { 1 }));
        var empty = new CompilationCache(FakeArtifactDevice.Code());
        var c = new CompilationCache(FakeArtifactDevice.Code(), [empty, stored]);
        Assert.True(c.TryGet("old", out _));
        Assert.True(c.TryGet("old", out _));
        c.Put("new", new byte[] { 2 });
        Assert.False(c.TryGet("missing", out _));
        Assert.Equal(["new", "old"], c.Entries.Keys.Order());
        Assert.Equal(1, c.FallbackHits);
        c.Put("old", new byte[] { 3 }); // replaced: no longer what the stored set holds
        Assert.Equal(0, c.FallbackHits);
    }

    /// <summary>A load consults the stored set of each tier independently: code hits without binaries and the other way round.</summary>
    [Theory]
    [InlineData(null)]
    [InlineData(ArtifactTiers.Code)]
    [InlineData(ArtifactTiers.Binary)]
    public void LoadUsesTheStoredTiersThatMatch(string? strippedTier)
    {
        using var tmp = new TempDir();
        var p = tmp.File("tiers.slm");
        var code = Set(FakeArtifactDevice.Code(subgroup: 64), ("kernel/a", new byte[] { 1 }));
        var binary = Set(FakeArtifactDevice.Binary(), (FakeArtifactDevice.BinaryKey, new byte[] { 2 }));
        var otherDriver = Set(FakeArtifactDevice.Binary(driver: "2"), ("fake/other", new byte[] { 3 }));
        ModelPackage.Save(p, PackageTests.SmallDefinition(), [code, binary, otherDriver]);
        if (strippedTier is not null)
        {
            Assert.Equal(strippedTier == ArtifactTiers.Binary ? 2 : 1, ModelPackage.Strip(p, "fake", strippedTier));
        }

        var source = new ModelPackageImporter().Open(p);
        try
        {
            var device = new FakeArtifactDevice();
            Assert.Equal(strippedTier == ArtifactTiers.Code ? 0 : 1, CompilationArtifacts.Matching(source.CompilationCaches!, device, device.ArtifactTargets[0]).Count);
            var cache = new CompilationArtifacts(source.CompilationCaches).For(device)!;
            Assert.Equal(strippedTier != ArtifactTiers.Code, cache.TryGet("kernel/a", out _));
            Assert.Equal(strippedTier != ArtifactTiers.Binary, cache.TryGet(FakeArtifactDevice.BinaryKey, out _));
            Assert.False(cache.TryGet("fake/other", out _)); // stored for another driver
        }
        finally
        {
            Dispose(source);
        }
    }

    [Fact]
    public void StripAndCompactRemoveDeadSpace()
    {
        using var tmp = new TempDir();
        var p = tmp.File("c.slm");
        var bulk = new byte[100_000];
        new Random(3).NextBytes(bulk);
        ModelPackage.Save(p, PackageTests.SmallDefinition(), [
            Set(FakeArtifactDevice.Code(), ("kernel/1", bulk)),
            Set(FakeArtifactDevice.Binary(), (FakeArtifactDevice.BinaryKey, bulk)),
            Set(FakeArtifactDevice.Code(provider: "cuda"), ("kernel/c", new byte[] { 5 })),
        ]);
        Assert.Equal(0, ModelPackageReader.Open(p).ReclaimableBytes);
        ModelPackage.AddArtifacts(p, [Set(FakeArtifactDevice.Code(), ("kernel/1", new byte[] { 1 }))]);
        Assert.Equal(1, ModelPackage.Strip(p, "fake", ArtifactTiers.Binary));
        Assert.Equal(0, ModelPackage.Strip(p, "nobody"));

        var before = ModelPackageReader.Open(p);
        Assert.True(before.ReclaimableBytes > 2 * bulk.Length, $"{before.ReclaimableBytes} reclaimable");
        long saved = ModelPackage.Compact(p);
        Assert.Equal(before.ReclaimableBytes, saved);
        Assert.Equal(before.FileSize - saved, new FileInfo(p).Length);

        var after = ModelPackageReader.Open(p);
        Assert.Equal(0, after.ReclaimableBytes);
        Assert.Equal(["cuda", "fake"], after.ArtifactSets.Select(s => s.Provider).Order());
        var source = new ModelPackageImporter().Open(p);
        try
        {
            Assert.Equal(PackageTests.SmallDefinition().Weights["w"].ReadAll(), source.Weights["w"].ReadAll());
            var sets = source.CompilationCaches!.ArtifactSets;
            Assert.Equal(new byte[] { 1 }, Get(sets.Single(s => s.Descriptor[ArtifactDescriptorKeys.Provider] == "fake").Entries, "kernel/1"));
            Assert.Equal(new byte[] { 5 }, Get(sets.Single(s => s.Descriptor[ArtifactDescriptorKeys.Provider] == "cuda").Entries, "kernel/c"));
        }
        finally
        {
            Dispose(source);
        }

        Assert.Equal(2, ModelPackage.Strip(p));
        Assert.Empty(ModelPackageReader.Open(p).ArtifactSets);
    }

    /// <summary>Version 1 packages (kernel sets in KSET sections) open, and their sets serve devices with the same v1 target.</summary>
    [Fact]
    public void Version1PackagesStayReadable()
    {
        using var tmp = new TempDir();
        var p = tmp.File("v1.slm");
        var device = new FakeArtifactDevice();
        string v1Target = ((IComputeDevice)device).CompilationTarget;
        ModelPackage.Save(p, PackageTests.SmallDefinition(), []);
        AppendVersion1KernelSet(p, v1Target, ("kernel/k", new byte[] { 4, 5 }), (FakeArtifactDevice.BinaryKey, new byte[] { 6 }));
        AppendVersion1KernelSet(p, "fake|Other GPU|1|", ("kernel/z", new byte[] { 9 }));

        var info = ModelPackageReader.Open(p);
        Assert.Equal(1, info.Version);
        Assert.Equal(["DEFN", "WIDX", "WDAT", "KSET", "KSET"], info.Sections.Select(s => s.Kind));
        var legacy = info.ArtifactSets[0];
        Assert.True(legacy.IsLegacy);
        Assert.Equal((ArtifactTiers.Mixed, "fake", v1Target), (legacy.Tier, legacy.Provider, legacy.Id));
        Assert.Equal([new ArtifactEntryInfo(FakeArtifactDevice.BinaryKey, 1), new ArtifactEntryInfo("kernel/k", 2)], legacy.Entries);

        var source = new ModelPackageImporter().Open(p);
        try
        {
            var cache = new CompilationArtifacts(source.CompilationCaches).For(device)!;
            Assert.Equal(new byte[] { 4, 5 }, Get(cache, "kernel/k"));
            Assert.Equal(new byte[] { 6 }, Get(cache, FakeArtifactDevice.BinaryKey));
            Assert.False(cache.TryGet("kernel/z", out _));

            // Appending converts the file to version 2 while it is open; the KSET sections stay.
            ModelPackage.AddArtifacts(p, [Set(FakeArtifactDevice.Code(), ("kernel/new", new byte[] { 1 }))]);
        }
        finally
        {
            Dispose(source);
        }

        info = ModelPackageReader.Open(p);
        Assert.Equal(ModelPackage.Version, info.Version);
        Assert.Equal(3, info.ArtifactSets.Count);
        Assert.Equal(2, ModelPackage.Strip(p, tier: ArtifactTiers.Mixed));
        Assert.Equal(ArtifactTiers.Code, Assert.Single(ModelPackageReader.Open(p).ArtifactSets).Tier);
    }

    [Fact]
    public void InspectionDescribesDefinitionWeightsAndArtifactsWithoutLoading()
    {
        using var tmp = new TempDir();
        var p = tmp.File("i.slm");
        ModelPackage.Save(p, PackageTests.SmallDefinition(), [Set(FakeArtifactDevice.Code(), ("kernel/b", new byte[10]), ("kernel/a", new byte[3]))]);

        var info = ModelPackageReader.Open(p);
        Assert.Equal(p, info.Path);
        Assert.Equal(new FileInfo(p).Length, info.FileSize);
        Assert.Equal(["DEFN", "WIDX", "WDAT", "ARTS"], info.Sections.Select(s => s.Kind));
        Assert.Equal(info.ArtifactSets[0].Id, info.Sections[3].Name);

        var w = Assert.Single(info.Weights);
        Assert.Equal(("w", "f32", 4L * 8 * 4), (w.Name, w.Type.Name, w.ByteSize));
        Assert.Equal([4L, 8L], w.Shape.Dims.Select(d => d.Fixed));
        Assert.Equal(0, w.Offset % ModelPackage.WeightAlignment);
        Assert.Equal(w.ByteSize, info.WeightBytes);

        var set = Assert.Single(info.ArtifactSets);
        Assert.False(set.IsLegacy);
        Assert.Equal(("fake", ArtifactTiers.Code, "spirv", 2, 13L), (set.Provider, set.Tier, set.Kind, set.EntryCount, set.TotalBytes));
        Assert.Equal(["kernel/a", "kernel/b"], set.Entries.Select(e => e.Key));

        var definition = info.Definition;
        Assert.Equal(("test_arch", "small"), (definition.Architecture, definition.Name));
        Assert.Equal("kc", Assert.Single(definition.States).Name);
        var graph = Assert.Single(definition.Graphs);
        Assert.Equal(("decode", 3, 1, 128L, 1), (graph.Name, graph.NodeCount, graph.WeightCount, graph.WeightBytes, graph.StateCount));
        Assert.Equal(["Add", "CacheWrite", "MatMul"], graph.OpHistogram.Select(kv => kv.Key));
        var input = Assert.Single(graph.Inputs);
        Assert.Equal(("x", "Embeddings"), (input.Name, input.Role));
        Assert.Equal("[T, 8]", input.Shape.ToString());
        Assert.Equal("y", Assert.Single(graph.Outputs).Name);
        Assert.Equal(["MatMul", "Add", "CacheWrite"], graph.Nodes.Select(n => n.Op));
        Assert.Equal("proj", graph.Nodes[0].Outputs[0].Name);
        Assert.Equal("layers.0", graph.Nodes[0].Group);

        Assert.Equal("[3: a, b, ü]", DefinitionInfo.FormatValue(definition.Metadata["tokenizer.tokens"]));
        Assert.Equal("[2: 1, 2]", DefinitionInfo.FormatValue(definition.Properties[WellKnown.EosTokenIds]));
        Assert.Equal("[3: 1, 2, ...]", DefinitionInfo.FormatValue(new[] { 1, 2, 3 }, maxItems: 2));
        Assert.Equal("abc... (5 chars)", DefinitionInfo.FormatValue("abcde", maxLength: 3));
        Assert.Equal("1.5", DefinitionInfo.FormatValue(1.5f));
    }

    /// <summary>Appends a version 1 KSET section (directory + 16-byte aligned blobs) and a table, and marks the file version 1.</summary>
    private static void AppendVersion1KernelSet(string path, string target, params (string Key, byte[] Data)[] entries)
    {
        using var fs = new FileStream(path, FileMode.Open, FileAccess.ReadWrite);
        var (_, sections) = ModelPackage.ReadHeader(fs);
        long start = PackageWriter.Align(fs.Seek(0, SeekOrigin.End), 16);
        PackageWriter.Pad(fs, start);
        var sorted = entries.OrderBy(e => e.Key, StringComparer.Ordinal).ToArray();
        int directoryLength = 4 + sorted.Sum(e => 4 + Encoding.UTF8.GetByteCount(e.Key) + 16);
        var w = new BinaryWriter(fs, Encoding.UTF8, leaveOpen: true);
        w.Write(sorted.Length);
        long blob = PackageWriter.Align(directoryLength, 16);
        var offsets = new List<long>();
        foreach (var (key, data) in sorted)
        {
            var name = Encoding.UTF8.GetBytes(key);
            w.Write(name.Length);
            w.Write(name);
            w.Write(blob);
            w.Write((long)data.Length);
            offsets.Add(blob);
            blob = PackageWriter.Align(blob + data.Length, 16);
        }
        w.Flush();
        for (int i = 0; i < sorted.Length; i++)
        {
            PackageWriter.Pad(fs, start + offsets[i]);
            fs.Write(sorted[i].Data);
        }

        var list = sections.Append(new PackageSection(PackageSectionKind.KernelSet, target, start, fs.Position - start)).ToList();
        PackageWriter.WriteTable(fs, list);
        Span<byte> version = stackalloc byte[4];
        BinaryPrimitives.WriteInt32LittleEndian(version, 1);
        fs.Seek(8, SeekOrigin.Begin);
        fs.Write(version);
    }
}
