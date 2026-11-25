using Sharplight.Hub;
using Sharplight.Studio.Services;
using Sharplight.Studio.Services.Library;
using Sharplight.Studio.Services.Maintenance;
using Sharplight.Studio.Tests.Fakes;

namespace Sharplight.Studio.Tests;

/// <summary>
/// Studio services whose library and maintenance are the real ones over a model store in a temporary directory (packages written with
/// <see cref="TestPackages"/>), with a fake runtime that says what is resident.
/// </summary>
internal sealed class MaintenanceFixture : IDisposable
{
    private readonly TestServices _services = new();

    public MaintenanceFixture()
    {
        Library = new ModelLibrary(_services.Settings, null!);
        Maintenance = new ModelMaintenanceService(_services.Settings, _services.Storage, _services.Runtime, Library, _services.Usage, _services.Precompiler);
        Services = _services.Services with { Library = Library, Maintenance = Maintenance };
    }

    public StudioServices Services { get; }

    public TestServices Test => _services;

    public ModelLibrary Library { get; }

    public ModelMaintenanceService Maintenance { get; }

    public ModelStore Store => new(_services.Settings.StoreRoot);

    public string Root => _services.Directory;

    /// <summary>Adds a stored model: a small package with precompiled artifacts (or the described one without any) and its manifest.</summary>
    public ModelManifest AddModel(string name, bool precompiled = true, DateTimeOffset? pulledAt = null)
    {
        var repository = $"tests/{name}";
        var directory = Store.GetModelDirectory(repository, "");
        Directory.CreateDirectory(directory);
        var package = Path.Combine(directory, ModelStore.PackageFileName);
        if (precompiled)
        {
            TestPackages.WriteSmall(package);
        }
        else
        {
            TestPackages.WriteDescribed(package);
        }

        var manifest = new ModelManifest
        {
            Repository = repository,
            Alias = name,
            Commit = "c0ffee",
            Format = ModelFormat.Slm,
            PackageBytes = new FileInfo(package).Length,
            PulledAt = pulledAt ?? DateTimeOffset.UtcNow,
            Directory = directory,
        };
        Store.Save(manifest);
        return manifest;
    }

    /// <summary>Marks the model as resident in the runtime.</summary>
    public FakeModel MakeResident(ModelManifest manifest) => _services.Runtime.Add(new FakeModel(manifest.Alias) { Path = manifest.PackagePath });

    public async Task<LibraryEntry> EntryAsync(ModelManifest manifest)
        => (await Library.ListAsync(CancellationToken.None)).Single(entry => entry.Path == manifest.PackagePath);

    public void Dispose() => _services.Dispose();
}
