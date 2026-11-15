using Sharplight.Hub;
using Sharplight.Package;
using Sharplight.Studio.Services.Settings;
using Sharplight.Vision;

namespace Sharplight.Studio.Services.Library;

/// <summary><see cref="IModelLibrary"/> over the <see cref="ModelStore"/> and the list of added files in the settings.</summary>
internal sealed class ModelLibrary : IModelLibrary
{

    private readonly ModelConverter _converter;
    private readonly ISettingsService _settings;

    public ModelLibrary(ISettingsService settings, ModelConverter converter)
    {
        _settings = settings;
        _converter = converter;
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
        await Task.Run(() =>
        {
            Directory.CreateDirectory(System.IO.Path.GetDirectoryName(System.IO.Path.GetFullPath(destination))!);
            var options = new ConversionOptions { Log = log.Report };
            _converter.Convert(source, destination, options, projector is null ? null : [new(AuxiliaryRoles.Mmproj, projector)]);
        }, cancellationToken).ConfigureAwait(false);
        await AddLocalAsync(destination).ConfigureAwait(false);
        return destination;
    }

    /// <inheritdoc/>
    public Task<LibraryEntry> ReloadAsync(LibraryEntry entry, CancellationToken cancellationToken)
    {
        return Task.Run(() => entry.Manifest is { } manifest ? FromManifest(manifest) : FromFile(entry.Path), cancellationToken);
    }

    /// <summary>
    /// A stored model. Size and precompiled sets come from the package itself, not the manifest, so they stay right when the package is
    /// precompiled again after the pull.
    /// </summary>
    private static LibraryEntry FromManifest(ModelManifest manifest)
    {
        var inspected = ModelInspector.Inspect(manifest.PackagePath);
        var file = new FileInfo(manifest.PackagePath);
        return new LibraryEntry
        {
            Id = manifest.ShortName,
            DisplayName = manifest.Name,
            Path = manifest.PackagePath,
            Architecture = manifest.Architecture ?? inspected.Architecture,
            Quantization = manifest.Tag.Length == 0 ? null : manifest.Tag,
            Format = manifest.Format.ToString(),
            SizeBytes = file.Exists ? file.Length : manifest.PackageBytes,
            Capabilities = inspected.Capabilities,
            TrainedContextLength = inspected.ContextLength,
            Kind = inspected.Kind,
            KindInferred = inspected.KindInferred,
            AddedAt = manifest.PulledAt,
            Precompiled = inspected.Precompiled,
            Manifest = manifest,
        };
    }

    private static LibraryEntry FromFile(string path)
    {
        var inspected = ModelInspector.Inspect(path);
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
            Architecture = inspected.Architecture,
            Format = directory ? "Directory" : System.IO.Path.GetExtension(path).TrimStart('.').ToUpperInvariant(),
            SizeBytes = size,
            Capabilities = inspected.Capabilities,
            TrainedContextLength = inspected.ContextLength,
            Kind = inspected.Kind,
            KindInferred = inspected.KindInferred,
            AddedAt = directory ? null : File.GetLastWriteTimeUtc(path),
            Precompiled = inspected.Precompiled,
        };
    }
}
