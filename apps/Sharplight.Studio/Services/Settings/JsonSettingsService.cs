using System.Text.Json;
using Sharplight.Studio.Models;

namespace Sharplight.Studio.Services.Settings;

/// <summary><see cref="ISettingsService"/> persisted as JSON; a missing or unreadable file gives the defaults.</summary>
internal sealed class JsonSettingsService : ISettingsService
{
    private readonly string _path;
    private readonly string? _commandLineStore;
    private readonly Lock _lock = new();
    private AppSettings _current;

    public JsonSettingsService(string path, string? commandLineStore)
    {
        _path = path;
        _commandLineStore = commandLineStore;
        _current = Load(path);
    }

    /// <inheritdoc/>
    public AppSettings Current
    {
        get
        {
            lock (_lock)
            {
                return _current;
            }
        }
    }

    /// <inheritdoc/>
    public string StoreRoot => StoreLocation.Resolve(_commandLineStore, Current.ModelsDirectory).Root;

    /// <inheritdoc/>
    public string StoreRootSource => StoreLocation.Resolve(_commandLineStore, Current.ModelsDirectory).Source;

    /// <inheritdoc/>
    public event EventHandler? Changed;

    /// <inheritdoc/>
    public void Update(Func<AppSettings, AppSettings> change)
    {
        lock (_lock)
        {
            _current = change(_current);
            Save(_path, _current);
        }

        Changed?.Invoke(this, EventArgs.Empty);
    }

    internal static AppSettings Load(string path)
    {
        try
        {
            return File.Exists(path) ? JsonSerializer.Deserialize(File.ReadAllText(path), StudioJsonContext.Default.AppSettings) ?? new AppSettings() : new AppSettings();
        }
        catch (Exception error) when (error is JsonException or IOException or UnauthorizedAccessException)
        {
            return new AppSettings();
        }
    }

    private static void Save(string path, AppSettings settings)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        var temporary = path + ".tmp";
        File.WriteAllText(temporary, JsonSerializer.Serialize(settings, StudioJsonContext.Default.AppSettings));
        File.Move(temporary, path, overwrite: true);
    }
}
