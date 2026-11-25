using System.Text.Json;

namespace Sharplight.Studio.Services.Maintenance;

/// <summary><see cref="IModelUsageStore"/> over a JSON file that is rewritten atomically (temporary file, then rename) on every change.</summary>
internal sealed class JsonModelUsageStore : IModelUsageStore
{
    private static readonly StringComparer s_comparer = OperatingSystem.IsWindows() || OperatingSystem.IsMacOS() ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal;

    private readonly string _file;
    private readonly Lock _gate = new();
    private readonly Dictionary<string, DateTimeOffset> _lastUsed;

    public JsonModelUsageStore(string file)
    {
        _file = file;
        _lastUsed = new Dictionary<string, DateTimeOffset>(Read(file), s_comparer);
    }

    /// <inheritdoc/>
    public DateTimeOffset? LastUsed(string path)
    {
        lock (_gate)
        {
            return _lastUsed.TryGetValue(System.IO.Path.GetFullPath(path), out var when) ? when : null;
        }
    }

    /// <inheritdoc/>
    public void Record(string path, DateTimeOffset when)
    {
        lock (_gate)
        {
            _lastUsed[System.IO.Path.GetFullPath(path)] = when;
            Save();
        }
    }

    /// <inheritdoc/>
    public void Forget(string path)
    {
        lock (_gate)
        {
            if (_lastUsed.Remove(System.IO.Path.GetFullPath(path)))
            {
                Save();
            }
        }
    }

    private static Dictionary<string, DateTimeOffset> Read(string file)
    {
        try
        {
            return File.Exists(file) ? JsonSerializer.Deserialize(File.ReadAllText(file), StudioJsonContext.Default.ModelUsageFile)?.LastUsed ?? [] : [];
        }
        catch (Exception error) when (error is JsonException or IOException or UnauthorizedAccessException)
        {
            return [];
        }
    }

    private void Save()
    {
        try
        {
            Directory.CreateDirectory(System.IO.Path.GetDirectoryName(_file)!);
            var temporary = _file + ".tmp";
            File.WriteAllText(temporary, JsonSerializer.Serialize(new ModelUsageFile { LastUsed = _lastUsed }, StudioJsonContext.Default.ModelUsageFile));
            File.Move(temporary, _file, overwrite: true);
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            // Last-used times are a convenience; a read-only data folder must not break loading.
        }
    }
}
