using Sharplight.ModelFeed.Client;
using Sharplight.ModelFeed.Publishing;
using Sharplight.ModelFeed.Signing;

namespace Sharplight.ModelFeed.Tests;

/// <summary>A generated feed in a scratch folder: test root and feed keys, packages with fake package files.</summary>
internal sealed class TestFeed : IDisposable
{
    public const long ChunkSize = 4096;

    public TestFeed()
    {
        Root = Path.Combine(Environment.GetEnvironmentVariable("SHARPLIGHT_WORK") ?? Path.GetTempPath(), "modelfeed-tests", Guid.NewGuid().ToString("N"));
        Folder = Path.Combine(Root, "feed");
        Staging = Path.Combine(Root, "staging");
        Directory.CreateDirectory(Folder);
        RootKey = FeedSigningKey.Generate("test root");
        FeedKey = FeedSigningKey.Generate("test feed");
        Time = new ManualTime(new DateTimeOffset(2026, 11, 17, 10, 0, 0, TimeSpan.Zero));
        Publisher = new FeedPublisher(Time);
        FeedPublisher.WriteKeySet(Folder, KeySet(FeedKey), RootKey);
    }

    public string Root { get; }

    public string Folder { get; }

    public string Staging { get; }

    public FeedSigningKey RootKey { get; }

    public FeedSigningKey FeedKey { get; }

    public ManualTime Time { get; }

    public FeedPublisher Publisher { get; }

    public static FeedKeySet KeySet(params FeedSigningKey[] keys) => new()
    {
        Version = 1,
        Issued = new DateTimeOffset(2026, 1, 1, 0, 0, 0, TimeSpan.Zero),
        Keys = [.. keys.Select(key => new FeedKeySetEntry { KeyId = key.KeyId, PublicKey = key.PublicKey.Base64 })],
    };

    public FeedClient Client(IFeedStateStore? state = null, bool allowExpired = false, IFeedTransport? transport = null) =>
        new(transport ?? new FolderFeedTransport(Folder), new FeedClientOptions { TrustedRoots = [RootKey.PublicKey], AllowExpiredIndex = allowExpired }, state, Time);

    /// <summary>Stages a package version with <paramref name="size"/> pseudo-random bytes and returns its folder.</summary>
    public async Task<string> StageAsync(string id, string version, int size = 10_000, IReadOnlyList<PackageDependency>? dependencies = null, string? server = null, byte seed = 1)
    {
        var folder = Path.Combine(Staging, $"{id}-{version}");
        Directory.CreateDirectory(folder);
        var name = FeedLayout.PackageFileName(id, version);
        var bytes = new byte[size];
        new Random(seed + id.GetHashCode(StringComparison.Ordinal)).NextBytes(bytes);
        await File.WriteAllBytesAsync(Path.Combine(folder, name), bytes);
        var file = await PackageFileHasher.DescribeAsync(Path.Combine(folder, name), name, ChunkSize);
        var manifest = new PackageManifest
        {
            Id = id,
            Version = version,
            Display = new PackageDisplay { Name = id },
            Publisher = new PackagePublisher { Name = "Test", CountryOfOrigin = "CH" },
            License = new PackageLicense { Spdx = "mit" },
            Model = new PackageModel { Kind = "text-generation", Tasks = [PackageTasks.Chat] },
            Requirements = new PackageRequirements { Server = server, Engine = ">=0.1.0", PackageFormat = 3 },
            Dependencies = dependencies ?? [],
            Files = [file],
        };
        await File.WriteAllBytesAsync(Path.Combine(folder, FeedLayout.ManifestFileName), FeedJson.Write(manifest, FeedJsonContext.Default.PackageManifest));
        return folder;
    }

    public async Task PublishAsync(params string[] folders) => await Publisher.PublishAsync(Folder, folders, FeedKey);

    public string PathOf(string relative) => Path.Combine(Folder, relative.Replace('/', Path.DirectorySeparatorChar));

    public void Dispose()
    {
        RootKey.Dispose();
        FeedKey.Dispose();
        if (Directory.Exists(Root))
        {
            Directory.Delete(Root, recursive: true);
        }
    }
}
