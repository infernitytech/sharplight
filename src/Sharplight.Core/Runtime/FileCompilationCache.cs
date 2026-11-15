using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text;

namespace Sharplight.Runtime;

/// <summary>
/// The entries of persistent artifact sets (see <see cref="ArtifactStore"/>) in a directory shared by processes: one file per entry,
/// named by a hash of the key, holding the key, the data and its SHA-256, so a truncated, corrupted or colliding file is detected and
/// ignored (and removed) instead of trusted. Keys must identify the content completely (kernel source, macros, compiler and target, as
/// the content-addressed keys of the code tier do). Writes go to a temporary file that atomically replaces the entry; hits refresh the
/// file time, and when the directory grows past <see cref="MaxBytes"/> the least recently used files are deleted. Safe under concurrent
/// processes: every file is complete or absent, and a file deleted by another process is just a miss. Thread-safe.
/// </summary>
/// <remarks>
/// A cache may be split into partitions (<see cref="Partition"/>, one per artifact set of a store): each partition is a subdirectory,
/// and the size cap and eviction cover the whole tree. Diagnostic environment switch: <c>SHARPLIGHT_COMPILATION_CACHE</c> = a directory
/// (used instead of the per-user default) or <c>off</c> (no persistent store), see <see cref="ResolveDirectory"/>.
/// </remarks>
public sealed class FileCompilationCache : ICompilationCache
{
    /// <summary>Default <see cref="MaxBytes"/>: 2 GiB.</summary>
    public const long DefaultMaxBytes = 2L << 30;

    /// <summary>Name of the environment variable that relocates or disables the cache.</summary>
    public const string EnvironmentVariable = "SHARPLIGHT_COMPILATION_CACHE";

    private const string Extension = ".slc";
    private const int FormatVersion = 1;
    private const int HashSize = 32;
    private static ReadOnlySpan<byte> Magic => "SLCC"u8;
    private static readonly TimeSpan s_touchInterval = TimeSpan.FromHours(1);
    private static readonly TimeSpan s_staleTemporary = TimeSpan.FromHours(1);

    private readonly Lock _lock = new();
    /// <summary>The cache that owns the size cap of the tree (this one, or the parent of a partition).</summary>
    private readonly FileCompilationCache _owner;
    private long _approximateBytes = -1;
    private long _writtenSinceScan;

    /// <summary>Opens (and creates) the cache in <paramref name="directory"/>.</summary>
    /// <param name="directory">The cache directory; shared by processes.</param>
    /// <param name="maxBytes">Size cap of the directory in bytes.</param>
    public FileCompilationCache(string directory, long maxBytes = DefaultMaxBytes)
    {
        ArgumentException.ThrowIfNullOrEmpty(directory);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxBytes);
        Directory = directory;
        MaxBytes = maxBytes;
        _owner = this;
        System.IO.Directory.CreateDirectory(directory);
    }

    private FileCompilationCache(FileCompilationCache owner, string directory)
    {
        Directory = directory;
        MaxBytes = owner.MaxBytes;
        _owner = owner;
        System.IO.Directory.CreateDirectory(directory);
    }

    /// <summary>A partition in the subdirectory <paramref name="name"/>, sharing this cache's size cap and eviction.</summary>
    /// <param name="name">The subdirectory name (a file name, e.g. an artifact set id).</param>
    public FileCompilationCache Partition(string name)
    {
        ArgumentException.ThrowIfNullOrEmpty(name);
        return new FileCompilationCache(_owner, Path.Combine(Directory, name));
    }

    /// <summary>The cache directory.</summary>
    public string Directory { get; }

    /// <summary>Size cap of the directory in bytes; past it the least recently used files are deleted.</summary>
    public long MaxBytes { get; }

    /// <summary>Entries found intact.</summary>
    public int Hits => Volatile.Read(ref _hits);

    /// <summary>Entry files found damaged or written for another key, ignored and removed.</summary>
    public int Rejected => Volatile.Read(ref _rejected);

    private int _hits;
    private int _rejected;

    /// <summary>
    /// The directory of the persistent cache: the <see cref="EnvironmentVariable"/> directory when set (<see langword="null"/> when it
    /// says <c>off</c>, <c>0</c> or <c>false</c>), else <paramref name="defaultDirectory"/>.
    /// </summary>
    /// <param name="defaultDirectory">The directory to use without the environment variable, or <see langword="null"/> for none.</param>
    public static string? ResolveDirectory(string? defaultDirectory)
    {
        var configured = Environment.GetEnvironmentVariable(EnvironmentVariable);
        if (string.IsNullOrWhiteSpace(configured))
        {
            return defaultDirectory;
        }

        return configured.Trim().ToLowerInvariant() is "off" or "0" or "false" or "none" ? null : configured.Trim();
    }

    /// <inheritdoc/>
    public bool TryGet(string key, out ReadOnlyMemory<byte> data)
    {
        data = default;
        var path = PathOf(key);
        byte[] file;
        try
        {
            file = File.ReadAllBytes(path);
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            // Absent, or removed by another process meanwhile: a miss.
            return false;
        }

        if (!TryDecode(file, key, out data))
        {
            Interlocked.Increment(ref _rejected);
            TryDelete(path);
            return false;
        }

        Interlocked.Increment(ref _hits);
        Touch(path);
        return true;
    }

    /// <inheritdoc/>
    public void Put(string key, ReadOnlyMemory<byte> data)
    {
        var path = PathOf(key);
        var temporary = $"{path}.{Environment.ProcessId}.{Environment.CurrentManagedThreadId}.tmp";
        var encoded = Encode(key, data.Span);
        try
        {
            File.WriteAllBytes(temporary, encoded);
            File.Move(temporary, path, overwrite: true);
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            // Read-only or full disk: the cache is only an accelerator.
            TryDelete(temporary);
            return;
        }

        _owner.OnWritten(encoded.Length);
    }

    /// <summary>Removes the entry of <paramref name="key"/> (e.g. data the consumer found unusable); absent entries are ignored.</summary>
    /// <param name="key">The entry key.</param>
    public void Remove(string key) => TryDelete(PathOf(key));

    private void OnWritten(long bytes)
    {
        if (Interlocked.Add(ref _writtenSinceScan, bytes) > MaxBytes / 16 || Volatile.Read(ref _approximateBytes) < 0)
        {
            Trim();
        }
        else if (Interlocked.Add(ref _approximateBytes, bytes) > MaxBytes)
        {
            Trim();
        }
    }

    /// <summary>
    /// Deletes the least recently used files of the whole tree (entries, and files of older cache layouts) until it is under 90% of
    /// <see cref="MaxBytes"/>, and temporary files left behind by processes that died while writing. Descriptor files (<c>*.json</c>)
    /// are kept.
    /// </summary>
    public void Trim()
    {
        if (_owner != this)
        {
            _owner.Trim();
            return;
        }

        lock (_lock)
        {
            Interlocked.Exchange(ref _writtenSinceScan, 0);
            var files = new List<FileInfo>();
            long total = 0;
            try
            {
                foreach (var file in new DirectoryInfo(Directory).EnumerateFiles("*", SearchOption.AllDirectories))
                {
                    if (file.Name.EndsWith(".json", StringComparison.Ordinal))
                    {
                        continue;
                    }

                    if (file.Name.EndsWith(".tmp", StringComparison.Ordinal))
                    {
                        if (file.LastWriteTimeUtc < DateTime.UtcNow - s_staleTemporary)
                        {
                            TryDelete(file.FullName);
                        }

                        continue;
                    }

                    files.Add(file);
                    total += file.Length;
                }
            }
            catch (Exception error) when (error is IOException or UnauthorizedAccessException)
            {
                return;
            }

            long target = MaxBytes / 10 * 9;
            foreach (var file in total > MaxBytes ? files.OrderBy(f => f.LastWriteTimeUtc) : Enumerable.Empty<FileInfo>())
            {
                if (total <= target)
                {
                    break;
                }

                if (TryDelete(file.FullName))
                {
                    total -= file.Length;
                }
            }

            Volatile.Write(ref _approximateBytes, total);
        }
    }

    private string PathOf(string key)
    {
        var hash = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(key)))[..40].ToLowerInvariant();
        return Path.Combine(Directory, hash + Extension);
    }

    private static byte[] Encode(string key, ReadOnlySpan<byte> data)
    {
        var keyBytes = Encoding.UTF8.GetBytes(key);
        var file = new byte[Magic.Length + sizeof(int) + sizeof(int) + keyBytes.Length + sizeof(long) + HashSize + data.Length];
        var span = file.AsSpan();
        Magic.CopyTo(span);
        int position = Magic.Length;
        BinaryPrimitives.WriteInt32LittleEndian(span[position..], FormatVersion);
        position += sizeof(int);
        BinaryPrimitives.WriteInt32LittleEndian(span[position..], keyBytes.Length);
        position += sizeof(int);
        keyBytes.CopyTo(span[position..]);
        position += keyBytes.Length;
        BinaryPrimitives.WriteInt64LittleEndian(span[position..], data.Length);
        position += sizeof(long);
        SHA256.HashData(data, span.Slice(position, HashSize));
        position += HashSize;
        data.CopyTo(span[position..]);
        return file;
    }

    private static bool TryDecode(byte[] file, string key, out ReadOnlyMemory<byte> data)
    {
        data = default;
        var span = file.AsSpan();
        int header = Magic.Length + sizeof(int) + sizeof(int);
        if (span.Length < header || !span[..Magic.Length].SequenceEqual(Magic)
            || BinaryPrimitives.ReadInt32LittleEndian(span[Magic.Length..]) != FormatVersion)
        {
            return false;
        }

        int keyLength = BinaryPrimitives.ReadInt32LittleEndian(span[(Magic.Length + sizeof(int))..]);
        if (keyLength < 0 || span.Length - header < keyLength + sizeof(long) + HashSize)
        {
            return false;
        }

        // The file name is a hash of the key: the stored key must be the one asked for.
        if (!span.Slice(header, keyLength).SequenceEqual(Encoding.UTF8.GetBytes(key)))
        {
            return false;
        }

        int position = header + keyLength;
        long length = BinaryPrimitives.ReadInt64LittleEndian(span[position..]);
        position += sizeof(long);
        var hash = span.Slice(position, HashSize);
        position += HashSize;
        if (length != span.Length - position)
        {
            return false;
        }

        Span<byte> actual = stackalloc byte[HashSize];
        SHA256.HashData(span[position..], actual);
        if (!actual.SequenceEqual(hash))
        {
            return false;
        }

        data = file.AsMemory(position);
        return true;
    }

    /// <summary>Marks the file recently used (at most once per <see cref="s_touchInterval"/>, so hits stay cheap).</summary>
    private static void Touch(string path)
    {
        try
        {
            var now = DateTime.UtcNow;
            if (File.GetLastWriteTimeUtc(path) < now - s_touchInterval)
            {
                File.SetLastWriteTimeUtc(path, now);
            }
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            // Evicted by another process meanwhile.
        }
    }

    private static bool TryDelete(string path)
    {
        try
        {
            File.Delete(path);
            return true;
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            return false;
        }
    }
}
