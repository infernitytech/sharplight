namespace Sharplight.Hub.Tests;

public sealed class StoreTests : IDisposable
{
    private readonly TempDirectory _directory = new();
    private readonly ModelStore _store;

    public StoreTests()
    {
        _store = new ModelStore(_directory.Path);
    }

    public void Dispose() => _directory.Dispose();

    [Fact]
    public void ManifestRoundTrips()
    {
        var written = Add("Qwen/Qwen3-0.6B-GGUF", "Q8_0");

        var read = Assert.Single(_store.List());

        Assert.Equal(written.Directory, read.Directory);
        Assert.Equal("Qwen/Qwen3-0.6B-GGUF:Q8_0", read.Name);
        Assert.Equal("qwen3-0.6b:q8_0", read.ShortName);
        Assert.Equal(ModelFormat.Gguf, read.Format);
        Assert.Equal("c0ffee", read.Commit);
        Assert.Equal(1234, read.PackageBytes);
        Assert.Equal("Qwen3-0.6B-Q8_0.gguf", Assert.Single(read.SourceFiles).Path);
        Assert.Equal(["vulkan.code.x"], read.Precompiled);
        Assert.Equal(written.PulledAt, read.PulledAt);
        Assert.Equal(Path.Combine(written.Directory, ModelStore.PackageFileName), read.PackagePath);
        var json = File.ReadAllText(Path.Combine(written.Directory, ModelStore.ManifestFileName));
        Assert.Contains("\"format\": \"Gguf\"", json, StringComparison.Ordinal);
    }

    [Fact]
    public void FindsByFullNameAliasAndTag()
    {
        Add("Qwen/Qwen3-0.6B-GGUF", "Q8_0");
        Add("Qwen/Qwen3-0.6B", "");

        Assert.Equal("Qwen/Qwen3-0.6B-GGUF:Q8_0", _store.Find("qwen/qwen3-0.6b-gguf:q8_0")?.Name);
        Assert.Equal("Qwen/Qwen3-0.6B-GGUF:Q8_0", _store.Find("Qwen/Qwen3-0.6B-GGUF")?.Name);
        Assert.Equal("Qwen/Qwen3-0.6B-GGUF:Q8_0", _store.Find("qwen3-0.6b:q8_0")?.Name);
        Assert.Equal("Qwen/Qwen3-0.6B-GGUF:Q8_0", _store.Find("Qwen/Qwen3-0.6B-GGUF:Qwen3-0.6B-Q8_0.gguf")?.Name);
        Assert.Equal("Qwen/Qwen3-0.6B", _store.Find("Qwen/Qwen3-0.6B")?.Name);
        Assert.Equal("Qwen/Qwen3-0.6B", _store.Find("qwen3-0.6b:default")?.Name);
        Assert.Null(_store.Find("Qwen/Qwen3-0.6B-GGUF:Q4_K_M"));
        Assert.Null(_store.Find("other"));
    }

    [Fact]
    public void AmbiguousNameListsCandidates()
    {
        Add("Qwen/Qwen3-0.6B-GGUF", "Q8_0");
        Add("Qwen/Qwen3-0.6B-GGUF", "Q4_K_M");

        var error = Assert.Throws<HubException>(() => _store.Find("qwen3-0.6b"));

        Assert.Contains("Qwen/Qwen3-0.6B-GGUF:Q4_K_M", error.Message, StringComparison.Ordinal);
        Assert.Contains("Qwen/Qwen3-0.6B-GGUF:Q8_0", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void DirectoryWithoutManifestIsNotAModel()
    {
        var directory = _store.GetModelDirectory("owner/half", "Q8_0");
        Directory.CreateDirectory(Path.Combine(directory, ModelStore.SourceDirectoryName));
        File.WriteAllText(Path.Combine(directory, ModelStore.PendingPackageFileName), "partial");

        Assert.Empty(_store.List());
        Assert.Null(_store.Find("owner/half"));
    }

    [Fact]
    public void RemoveDeletesModelAndEmptyParents()
    {
        var kept = Add("owner/model", "Q8_0");
        var removed = Add("owner/model", "Q4_K_M");

        _store.Remove(removed);

        Assert.False(Directory.Exists(removed.Directory));
        Assert.True(Directory.Exists(kept.Directory));
        _store.Remove(kept);
        Assert.False(Directory.Exists(Path.Combine(_directory.Path, "owner")));
        Assert.True(Directory.Exists(_directory.Path));
    }

    [Fact]
    public void TagDirectoriesAreSafeFileNames()
    {
        Assert.Equal(_directory.Combine("o", "r", "default"), _store.GetModelDirectory("o/r", ""));
        Assert.Equal(_directory.Combine("o", "r", "a_b"), _store.GetModelDirectory("o/r", "a/b"));
        Assert.Equal(_directory.Combine("o", "r", "default"), _store.GetModelDirectory("o/r", ".."));
    }

    [Fact]
    public void SecondPullLockFails()
    {
        var directory = _store.GetModelDirectory("o/r", "Q8_0");
        using var first = ModelStore.Lock(directory, "o/r:Q8_0");

        var error = Assert.Throws<HubException>(() => ModelStore.Lock(directory, "o/r:Q8_0"));

        Assert.Contains("Another pull", error.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("Qwen/Qwen3-0.6B-GGUF", "qwen3-0.6b")]
    [InlineData("bartowski/Qwen_Qwen3-0.6B-GGUF", "qwen_qwen3-0.6b")]
    [InlineData("onnx-community/Qwen3-0.6B-ONNX", "qwen3-0.6b")]
    [InlineData("Qwen/Qwen3-0.6B", "qwen3-0.6b")]
    public void AliasDropsFormatSuffix(string repository, string alias)
    {
        Assert.Equal(alias, ModelManifest.AliasFor(repository));
    }

    private ModelManifest Add(string repository, string tag)
    {
        var directory = _store.GetModelDirectory(repository, tag);
        Directory.CreateDirectory(directory);
        var manifest = new ModelManifest
        {
            Repository = repository,
            Tag = tag,
            Alias = ModelManifest.AliasFor(repository),
            Commit = "c0ffee",
            Format = tag.Length > 0 ? ModelFormat.Gguf : ModelFormat.Safetensors,
            PackageBytes = 1234,
            SourceFiles = [new ManifestFile { Path = tag.Length > 0 ? $"Qwen3-0.6B-{tag}.gguf" : "model.safetensors", Size = 99 }],
            Precompiled = ["vulkan.code.x"],
            PulledAt = new DateTimeOffset(2026, 11, 13, 12, 0, 0, TimeSpan.Zero),
            Directory = directory,
        };
        ModelStore.WriteManifest(manifest);
        return manifest;
    }
}
