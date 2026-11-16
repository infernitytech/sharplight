using Sharplight.Storage;

namespace Sharplight.Core.Tests.Storage;

/// <summary>A temporary home and package folder with an injected environment for <see cref="StorageLayout.Resolve"/>.</summary>
internal sealed class StorageTestEnvironment : IDisposable
{
    public StorageTestEnvironment()
    {
        Root = Directory.CreateTempSubdirectory("sharplight-storage-test-").FullName;
        Home = Combine("home");
        Application = Combine("app");
        Directory.CreateDirectory(Home);
        Directory.CreateDirectory(Application);
    }

    public string Root { get; }

    public string Home { get; }

    /// <summary>The executable's folder (<see cref="StorageEnvironment.ApplicationDirectory"/>).</summary>
    public string Application { get; set; }

    public Dictionary<string, string> Variables { get; } = new(StringComparer.Ordinal);

    public StoragePlatform Platform { get; set; } = StoragePlatform.Linux;

    /// <summary>Folders the simulated write probe refuses.</summary>
    public HashSet<string> ReadOnly { get; } = new(StringComparer.Ordinal);

    public StorageEnvironment Environment => new()
    {
        ApplicationDirectory = Application,
        HomeDirectory = Home,
        Platform = Platform,
        GetVariable = name => Variables.GetValueOrDefault(name),
        CanWrite = directory => !ReadOnly.Contains(directory) && StorageWriteProbe.CanWrite(directory),
    };

    public string Combine(params string[] parts) => Path.Combine([Root, .. parts]);

    public StorageLayout Resolve() => StorageLayout.Resolve(Environment);

    public void Dispose()
    {
        if (Directory.Exists(Root))
        {
            Directory.Delete(Root, recursive: true);
        }
    }
}
