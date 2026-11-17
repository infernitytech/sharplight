using System.Text;
using System.Text.Json;
using Sharplight.Storage;

namespace Sharplight.Speech;

/// <summary>
/// The saved voices of an application: one folder per voice in the layout's <c>voices</c> data folder (<see cref="StorageLayout.GetDataDirectory(string)"/>),
/// holding <c>voice.json</c> (name, architecture, reference text, where each tensor lies) and <c>tensors.bin</c> (the raw arrays). A voice is
/// the conditioning of one model family (<see cref="VoiceProfile.Architecture"/>); the library lists the voices of every family.
/// </summary>
public sealed class VoiceLibrary
{
    /// <summary>The folder name of the voices inside the data directory.</summary>
    public const string DirectoryName = "voices";

    private const string ManifestFile = "voice.json";
    private const string TensorFile = "tensors.bin";

    /// <summary>Opens the library of a storage layout (in its <c>voices</c> data folder).</summary>
    /// <param name="layout">Where the application keeps its files.</param>
    public VoiceLibrary(StorageLayout layout)
        : this(DirectoryOf(layout))
    {
    }

    /// <summary>Opens the library in a folder.</summary>
    /// <param name="directory">The folder.</param>
    public VoiceLibrary(string directory)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directory);
        Root = directory;
    }

    /// <summary>The folder that holds the voices.</summary>
    public string Root { get; }

    /// <summary>The names of the saved voices, in alphabetical order.</summary>
    /// <param name="architecture">Only the voices of this architecture family, or <see langword="null"/> for all.</param>
    public IReadOnlyList<string> List(string? architecture = null)
    {
        if (!Directory.Exists(Root))
        {
            return [];
        }

        var names = new List<string>();
        foreach (string folder in Directory.EnumerateDirectories(Root))
        {
            if (TryReadManifest(folder) is { } manifest && (architecture is null || manifest.Architecture == architecture))
            {
                names.Add(manifest.Name);
            }
        }

        names.Sort(StringComparer.OrdinalIgnoreCase);
        return names;
    }

    /// <summary>Whether a voice of that name is saved.</summary>
    /// <param name="name">The voice's name.</param>
    public bool Contains(string name) => File.Exists(Path.Combine(FolderOf(name), ManifestFile));

    /// <summary>Saves a voice (replacing one of the same name).</summary>
    /// <param name="voice">The voice.</param>
    public void Save(VoiceProfile voice)
    {
        ArgumentNullException.ThrowIfNull(voice);
        string folder = FolderOf(voice.Name);
        Directory.CreateDirectory(folder);
        var manifest = new VoiceProfileManifest
        {
            Name = voice.Name,
            Architecture = voice.Architecture,
            ReferenceText = voice.ReferenceText,
            Created = voice.Created,
            SampleRate = voice.SampleRate,
        };
        string temporary = Path.Combine(folder, TensorFile + ".tmp");
        using (var stream = File.Create(temporary))
        {
            long offset = 0;
            foreach (var (name, tensor) in voice.Tensors.OrderBy(pair => pair.Key, StringComparer.Ordinal))
            {
                stream.Write(tensor.Data);
                manifest.Tensors[name] = new VoiceTensorEntry { DataType = tensor.DataType, Shape = tensor.Shape, Offset = offset, Length = tensor.Data.Length };
                offset += tensor.Data.Length;
            }
        }

        File.Move(temporary, Path.Combine(folder, TensorFile), overwrite: true);
        File.WriteAllText(Path.Combine(folder, ManifestFile), JsonSerializer.Serialize(manifest, SpeechJsonContext.Default.VoiceProfileManifest), Encoding.UTF8);
    }

    /// <summary>Reads a saved voice.</summary>
    /// <param name="name">The voice's name.</param>
    /// <exception cref="FileNotFoundException">No voice of that name is saved.</exception>
    public VoiceProfile Load(string name)
    {
        string folder = FolderOf(name);
        var manifest = TryReadManifest(folder) ?? throw new FileNotFoundException($"No saved voice named '{name}'.", folder);
        using var stream = File.OpenRead(Path.Combine(folder, TensorFile));
        var tensors = new Dictionary<string, VoiceTensor>(StringComparer.Ordinal);
        foreach (var (key, entry) in manifest.Tensors)
        {
            var data = new byte[entry.Length];
            stream.Seek(entry.Offset, SeekOrigin.Begin);
            stream.ReadExactly(data);
            tensors[key] = new VoiceTensor(entry.DataType, entry.Shape, data);
        }

        return new VoiceProfile
        {
            Name = manifest.Name,
            Architecture = manifest.Architecture,
            ReferenceText = manifest.ReferenceText,
            Created = manifest.Created,
            SampleRate = manifest.SampleRate,
            Tensors = tensors,
        };
    }

    /// <summary>Deletes a saved voice; nothing happens when there is none.</summary>
    /// <param name="name">The voice's name.</param>
    /// <returns>Whether a voice was deleted.</returns>
    public bool Delete(string name)
    {
        string folder = FolderOf(name);
        if (!Directory.Exists(folder))
        {
            return false;
        }

        Directory.Delete(folder, recursive: true);
        return true;
    }

    /// <summary>Renames a saved voice.</summary>
    /// <param name="name">The current name.</param>
    /// <param name="newName">The new name.</param>
    /// <exception cref="IOException">A voice with the new name exists.</exception>
    public void Rename(string name, string newName)
    {
        if (Contains(newName) && !string.Equals(FolderOf(name), FolderOf(newName), StringComparison.Ordinal))
        {
            throw new IOException($"A voice named '{newName}' exists.");
        }

        var voice = Load(name).Renamed(newName);
        Save(voice);
        if (!string.Equals(FolderOf(name), FolderOf(newName), StringComparison.Ordinal))
        {
            Delete(name);
        }
    }

    private static string DirectoryOf(StorageLayout layout)
    {
        ArgumentNullException.ThrowIfNull(layout);
        return layout.GetDataDirectory(DirectoryName);
    }

    private string FolderOf(string name)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        var builder = new StringBuilder(name.Length);
        foreach (char c in name.Trim())
        {
            builder.Append(char.IsLetterOrDigit(c) || c is '-' or '_' ? c : '_');
        }

        return Path.Combine(Root, builder.ToString());
    }

    private static VoiceProfileManifest? TryReadManifest(string folder)
    {
        string path = Path.Combine(folder, ManifestFile);
        if (!File.Exists(path) || !File.Exists(Path.Combine(folder, TensorFile)))
        {
            return null;
        }

        try
        {
            return JsonSerializer.Deserialize(File.ReadAllText(path), SpeechJsonContext.Default.VoiceProfileManifest);
        }
        catch (JsonException)
        {
            return null;
        }
    }
}
