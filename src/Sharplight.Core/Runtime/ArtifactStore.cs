using System.Collections.Concurrent;
using System.Text.Json;

namespace Sharplight.Runtime;

/// <summary>
/// The second location of precompiled artifacts, next to the artifact sets stored in a model package: a per-user directory of artifact
/// sets with exactly the descriptors, keys, tiers and matching of the package's sets (one subdirectory per set, named by its id, with
/// its descriptor in <c>descriptor.json</c> and validated entry files, see <see cref="FileCompilationCache"/>). It covers what packages
/// cannot: models loaded straight from GGUF, Hugging Face or ONNX files, graphs built in code (tests), and kernel development (entries
/// are content addressed, so only changed kernels compile again). Loads consult the package's sets first, then the store, then
/// compile, and record what they compiled into the store (<see cref="CompilationArtifacts"/>); packages only change when a host
/// precompiles or converts explicitly. Size capped (least recently used entries are evicted) and safe under concurrent processes.
/// Thread-safe.
/// </summary>
public sealed class ArtifactStore : ICompilationCacheStore
{
    private const string DescriptorFileName = "descriptor.json";

    /// <summary>One store per directory in the process, so its size accounting is shared by every device using it.</summary>
    private static readonly ConcurrentDictionary<string, ArtifactStore> s_opened = new(StringComparer.Ordinal);

    private readonly FileCompilationCache _entries;
    private readonly ConcurrentDictionary<string, StoredArtifactSet> _sets = new(StringComparer.Ordinal);
    private readonly Lock _lock = new();
    private StoredArtifactSet[]? _snapshot;

    /// <summary>Opens (and creates) a store.</summary>
    /// <param name="directory">The store directory; shared by processes.</param>
    /// <param name="maxBytes">Size cap of the store in bytes.</param>
    public ArtifactStore(string directory, long maxBytes = FileCompilationCache.DefaultMaxBytes)
    {
        _entries = new FileCompilationCache(directory, maxBytes);
    }

    /// <summary>The store directory.</summary>
    public string Directory => _entries.Directory;

    /// <summary>The per-user store directory (<c>sharplight/artifacts</c> under the local application data folder).</summary>
    public static string UserDirectory => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData, Environment.SpecialFolderOption.Create), "sharplight", "artifacts");

    /// <summary>
    /// The store a device with <paramref name="options"/> uses: none when <see cref="DeviceOptions.UseCompilationCache"/> is off, else
    /// the one in <see cref="DeviceOptions.KernelCacheDirectory"/>, else the per-user store unless
    /// <see cref="FileCompilationCache.EnvironmentVariable"/> relocates or disables it. Opened once per directory and process.
    /// </summary>
    /// <param name="options">The device options.</param>
    public static ArtifactStore? For(DeviceOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        var directory = !options.UseCompilationCache ? null : options.KernelCacheDirectory ?? FileCompilationCache.ResolveDirectory(UserDirectory);
        return directory is null ? null : s_opened.GetOrAdd(Path.GetFullPath(directory), path => new ArtifactStore(path));
    }

    /// <summary>The sets in the store (their entries are read on demand).</summary>
    public IReadOnlyList<StoredArtifactSet> ArtifactSets
    {
        get
        {
            lock (_lock)
            {
                return _snapshot ??= Scan();
            }
        }
    }

    /// <summary>The set of <paramref name="target"/>'s exact descriptor, created when missing; what a device records into.</summary>
    /// <param name="target">An artifact target of a device.</param>
    public StoredArtifactSet SetFor(ArtifactTarget target)
    {
        ArgumentNullException.ThrowIfNull(target);
        return _sets.GetOrAdd(target.Id, _ =>
        {
            var set = Open(target.Id, target.Descriptor);
            WriteDescriptor(target.Id, target.Descriptor);
            lock (_lock)
            {
                _snapshot = null;
            }

            return set;
        });
    }

    /// <summary>
    /// The entries of <paramref name="target"/>'s set (<see cref="SetFor"/>) with the operations of the entry files, such as removing an
    /// entry the device found unusable.
    /// </summary>
    /// <param name="target">An artifact target of a device.</param>
    public FileCompilationCache EntriesFor(ArtifactTarget target) => (FileCompilationCache)SetFor(target).Entries;

    /// <summary>Stores <paramref name="set"/> (e.g. device code from <see cref="IExecutionProvider.CompileDeviceSpecificAsync"/>).</summary>
    /// <param name="set">The code set; empty sets are ignored.</param>
    public void Add(CodeSet set)
    {
        ArgumentNullException.ThrowIfNull(set);
        if (set.IsEmpty)
        {
            return;
        }

        WriteDescriptor(set.Id, set.Descriptor);
        var entries = _sets.GetOrAdd(set.Id, id => Open(id, set.Descriptor)).Entries;
        foreach (var (key, data) in set.Entries)
        {
            entries.Put(key, data);
        }

        lock (_lock)
        {
            _snapshot = null;
        }
    }

    private StoredArtifactSet[] Scan()
    {
        var sets = new List<StoredArtifactSet>();
        try
        {
            foreach (var directory in new DirectoryInfo(Directory).EnumerateDirectories())
            {
                if (_sets.TryGetValue(directory.Name, out var known))
                {
                    sets.Add(known);
                }
                else if (ReadDescriptor(directory.FullName) is { } descriptor && ArtifactDescriptor.Id(descriptor) == directory.Name)
                {
                    // A directory whose descriptor does not hash to its name is damaged or foreign: not consulted.
                    sets.Add(_sets.GetOrAdd(directory.Name, id => Open(id, descriptor)));
                }
            }
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            // An unreadable store is an empty one; it is only an accelerator.
        }

        return [.. sets];
    }

    private StoredArtifactSet Open(string id, IReadOnlyDictionary<string, string> descriptor)
        => new(new Dictionary<string, string>(descriptor, StringComparer.Ordinal), _entries.Partition(id));

    private void WriteDescriptor(string id, IReadOnlyDictionary<string, string> descriptor)
    {
        var directory = Path.Combine(Directory, id);
        var path = Path.Combine(directory, DescriptorFileName);
        if (File.Exists(path))
        {
            return;
        }

        var temporary = $"{path}.{Environment.ProcessId}.{Environment.CurrentManagedThreadId}.tmp";
        try
        {
            System.IO.Directory.CreateDirectory(directory);
            var json = JsonSerializer.Serialize(new Dictionary<string, string>(descriptor), ArtifactStoreJsonContext.Default.DictionaryStringString);
            File.WriteAllText(temporary, json);
            File.Move(temporary, path, overwrite: true);
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            // Read-only or full disk: the store is only an accelerator.
        }
    }

    private static Dictionary<string, string>? ReadDescriptor(string directory)
    {
        try
        {
            return JsonSerializer.Deserialize(File.ReadAllText(Path.Combine(directory, DescriptorFileName)), ArtifactStoreJsonContext.Default.DictionaryStringString);
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or JsonException)
        {
            return null;
        }
    }
}
