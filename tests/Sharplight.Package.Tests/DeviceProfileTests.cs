using Sharplight.Runtime;
using Xunit;

namespace Sharplight.Package.Tests;

public class DeviceProfileTests
{
    private static DeviceProfile Profile() => new()
    {
        Provider = "vulkan",
        Name = "Test GPU",
        Vendor = "Vendor",
        Kind = DeviceKind.IntegratedGpu,
        TotalMemory = 1L << 33,
        CodeTarget = "spirv_1_6",
        SubgroupSize = 64,
        MaxSharedMemory = 65536,
        Features = [DeviceFeatures.Float16, DeviceFeatures.MatrixF16, DeviceFeatures.SubgroupOps],
        SubmissionBudgetMs = 250,
        Properties = new Dictionary<string, string> { ["coopmat_shapes"] = "16x16x16:f16*f16+f32->f32" },
    };

    [Fact]
    public void ProfileRoundTripsThroughJson()
    {
        var profile = Profile();
        var json = profile.ToJson();
        Assert.Contains("\"kind\": \"IntegratedGpu\"", json);
        Assert.Contains("f16*f16+f32->f32", json); // written for people: no + escapes
        var copy = DeviceProfile.FromJson(json);
        Assert.Equal(json, copy.ToJson());
        Assert.Equal(profile.Features, copy.Features);
        Assert.Equal(DeviceKind.IntegratedGpu, copy.Kind);
        Assert.Equal(250, copy.SubmissionBudgetMs);
        Assert.Equal(DeviceProfile.CurrentFormatVersion, copy.FormatVersion);
    }

    [Fact]
    public void FeaturesCanBeLeftOut()
    {
        var reduced = Profile().WithoutFeatures([DeviceFeatures.MatrixF16], "no matrix");
        Assert.Equal([DeviceFeatures.Float16, DeviceFeatures.SubgroupOps], reduced.Features);
        Assert.Equal("no matrix", reduced.Name);
        Assert.Equal(64, reduced.SubgroupSize);
    }

    [Theory]
    [InlineData("not json")]
    [InlineData("null")]
    [InlineData("{\"provider\":\"vulkan\",\"name\":\"x\",\"codeTarget\":\"spirv_1_6\",\"subgroupSize\":0,\"maxSharedMemory\":1}")]
    [InlineData("{\"formatVersion\":99,\"provider\":\"vulkan\",\"name\":\"x\",\"codeTarget\":\"spirv_1_6\",\"subgroupSize\":32,\"maxSharedMemory\":1}")]
    [InlineData("{\"name\":\"x\",\"codeTarget\":\"spirv_1_6\",\"subgroupSize\":32,\"maxSharedMemory\":1}")]
    public void InvalidProfilesAreRejected(string json) => Assert.Throws<InvalidDataException>(() => DeviceProfile.FromJson(json));

    [Fact]
    public void ArtifactSetsAreReadBackForCopying()
    {
        using var tmp = new TempDir();
        var p = tmp.File("a.slm");
        var code = new CompilationCache(FakeArtifactDevice.Code());
        code.Put("kernel/1", new byte[] { 1, 2, 3 });
        var binary = new CompilationCache(FakeArtifactDevice.Binary());
        binary.Put(FakeArtifactDevice.BinaryKey, new byte[100]);
        ModelPackage.Save(p, PackageTests.SmallDefinition(), [code, binary]);

        var sets = ModelPackage.ReadArtifactSets(p);
        Assert.Equal(2, sets.Count);
        foreach (var original in new[] { code, binary })
        {
            var read = sets.Single(s => ArtifactDescriptor.Id(s.Descriptor) == ArtifactDescriptor.Id(original.Descriptor));
            Assert.Equal(original.Entries.Keys.Order(StringComparer.Ordinal), read.Entries.Keys.Order(StringComparer.Ordinal));
            foreach (var (key, data) in original.Entries)
            {
                Assert.Equal(data.ToArray(), read.Entries[key].ToArray());
            }
        }

        // Stored again with another model file, the sets are found under the same ids.
        var copy = tmp.File("b.slm");
        ModelPackage.Save(copy, PackageTests.SmallDefinition(), sets);
        Assert.Equal(
            ModelPackageReader.Open(p).ArtifactSets.Select(s => s.Id).Order(StringComparer.Ordinal),
            ModelPackageReader.Open(copy).ArtifactSets.Select(s => s.Id).Order(StringComparer.Ordinal));
    }
}
