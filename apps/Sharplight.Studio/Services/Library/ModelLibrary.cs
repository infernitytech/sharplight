using Sharplight.Hub;
using Sharplight.Package.Import;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.Services.Settings;

namespace Sharplight.Studio.Services.Library;

/// <summary><see cref="IModelLibrary"/> over the <see cref="ModelStore"/> and the list of added files in the settings.</summary>
internal sealed class ModelLibrary : IModelLibrary
{
    /// <summary>Auxiliary source role of a vision projector.</summary>
    private const string ProjectorRole = "mmproj";

    private readonly ISettingsService _settings;

    public ModelLibrary(ISettingsService settings)
    {
        _settings = settings;
    }

    /// <inheritdoc/>
    public Task<IReadOnlyList<LibraryEntry>> ListAsync(CancellationToken cancellationToken)
    {
        return Task.Run<IReadOnlyList<LibraryEntry>>(() =>
        {
            var entries = new List<LibraryEntry>();
            foreach (var manifest in new ModelStore(_settings.StoreRoot).List())
            {
                cancellationToken.ThrowIfCancellationRequested();
                entries.Add(FromManifest(manifest));
            }

            foreach (var path in _settings.Current.LocalModels)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (File.Exists(path) || Directory.Exists(path))
                {
                    entries.Add(FromFile(path));
                }
            }

            return entries;
        }, cancellationToken);
    }

    /// <inheritdoc/>
    public Task RemoveAsync(LibraryEntry entry)
    {
        if (entry.Manifest is { } manifest)
        {
            return Task.Run(() => new ModelStore(_settings.StoreRoot).Remove(manifest));
        }

        _settings.Update(settings => settings with { LocalModels = [.. settings.LocalModels.Where(path => path != entry.Path)] });
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public Task AddLocalAsync(string path)
    {
        var full = System.IO.Path.GetFullPath(path);
        _settings.Update(settings => settings.LocalModels.Contains(full) ? settings : settings with { LocalModels = [.. settings.LocalModels, full] });
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public async Task<string> ConvertAsync(string source, string? projector, string destination, IProgress<string> log, CancellationToken cancellationToken)
    {
        EngineBootstrap.Register();
        await Task.Run(() =>
        {
            var options = new ConversionOptions { Log = log.Report };
            ModelConverter.Convert(source, destination, options, projector is null ? null : [(ProjectorRole, projector)]);
        }, cancellationToken).ConfigureAwait(false);
        await AddLocalAsync(destination).ConfigureAwait(false);
        return destination;
    }

    private static LibraryEntry FromManifest(ModelManifest manifest)
    {
        var (capabilities, context, architecture) = ModelInspector.Inspect(manifest.PackagePath);
        return new LibraryEntry
        {
            Id = manifest.ShortName,
            DisplayName = manifest.Name,
            Path = manifest.PackagePath,
            Architecture = manifest.Architecture ?? architecture,
            Quantization = manifest.Tag.Length == 0 ? null : manifest.Tag,
            Format = manifest.Format.ToString(),
            SizeBytes = manifest.PackageBytes,
            Capabilities = capabilities,
            TrainedContextLength = context,
            AddedAt = manifest.PulledAt,
            Precompiled = manifest.Precompiled,
            Manifest = manifest,
        };
    }

    private static LibraryEntry FromFile(string path)
    {
        var (capabilities, context, architecture) = ModelInspector.Inspect(path);
        bool directory = Directory.Exists(path);
        long size = directory
            ? new DirectoryInfo(path).EnumerateFiles("*", SearchOption.AllDirectories).Sum(file => file.Length)
            : new FileInfo(path).Length;
        var name = System.IO.Path.GetFileName(path.TrimEnd(System.IO.Path.DirectorySeparatorChar));
        return new LibraryEntry
        {
            Id = System.IO.Path.GetFileNameWithoutExtension(name).ToLowerInvariant(),
            DisplayName = name,
            Path = path,
            Architecture = architecture,
            Format = directory ? "Directory" : System.IO.Path.GetExtension(path).TrimStart('.').ToUpperInvariant(),
            SizeBytes = size,
            Capabilities = capabilities,
            TrainedContextLength = context,
            AddedAt = directory ? null : File.GetLastWriteTimeUtc(path),
        };
    }
}
