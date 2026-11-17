using System.Globalization;
using System.Text.Json;

namespace Sharplight.Studio.Services.Gallery;

/// <summary>
/// The generated images on disk: <c>&lt;folder&gt;/yyyy-MM-dd/&lt;id&gt;.png</c> with <c>&lt;id&gt;.json</c> (a <see cref="GalleryEntry"/>) next to it.
/// The folder lives in the data folder of the active storage mode. There is no index to keep in step: the sidecars are the index, so
/// copying, deleting or adding files by hand just works. All calls are meant for worker threads.
/// </summary>
internal sealed class ImageGalleryStore
{
    private readonly string _folder;

    /// <param name="folder">The gallery folder (<see cref="Sharplight.Studio.Services.Pages.IPageStorage.GenerationsDirectory"/>).</param>
    public ImageGalleryStore(string folder) => _folder = folder;

    /// <summary>The gallery folder.</summary>
    public string Folder => _folder;

    /// <summary>The absolute path of an entry's PNG.</summary>
    public string PathOf(GalleryEntry entry) => Path.Combine(_folder, entry.File.Replace('/', Path.DirectorySeparatorChar));

    /// <summary>The absolute path of an entry's video file (it need not exist: see <see cref="GalleryEntry.Video"/>).</summary>
    public string PathOfVideo(GalleryEntry entry) => Path.Combine(Path.GetDirectoryName(PathOf(entry))!, entry.Video);

    /// <summary>The folder of an entry's PNG (what "open containing folder" shows).</summary>
    public string DirectoryOf(GalleryEntry entry) => Path.GetDirectoryName(PathOf(entry)) ?? _folder;

    /// <summary>A new unique id.</summary>
    public static string NewId() => DateTime.Now.ToString("yyyyMMdd-HHmmssfff-", CultureInfo.InvariantCulture) + Guid.NewGuid().ToString("N")[..6];

    /// <summary>Writes the PNG and its sidecar (the PNG first, so an interrupted write leaves no sidecar without an image).</summary>
    /// <param name="entry">The metadata; its <see cref="GalleryEntry.Id"/> and <see cref="GalleryEntry.Created"/> name the files.</param>
    /// <param name="png">The PNG bytes.</param>
    /// <returns>The entry as stored (with <see cref="GalleryEntry.File"/>).</returns>
    /// <param name="video">The encoded video stored next to the PNG, or null (<see cref="GalleryEntry.Video"/> then names it).</param>
    public GalleryEntry Add(GalleryEntry entry, byte[] png, byte[]? video = null)
    {
        string day = entry.Created.LocalDateTime.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture);
        string id = entry.Id.Length > 0 ? entry.Id : NewId();
        var stored = entry with { Id = id, File = day + "/" + id + ".png", Video = video is null ? "" : entry.Video };
        string path = PathOf(stored);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        if (video is not null)
        {
            File.WriteAllBytes(PathOfVideo(stored), video);
        }

        File.WriteAllBytes(path, png);
        string sidecar = Path.ChangeExtension(path, ".json");
        File.WriteAllText(sidecar + ".tmp", JsonSerializer.Serialize(stored, GalleryJsonContext.Default.GalleryEntry));
        File.Move(sidecar + ".tmp", sidecar, overwrite: true);
        return stored;
    }

    /// <summary>Reads every entry whose PNG still exists, newest first. Damaged sidecars are skipped.</summary>
    /// <param name="cancellation">Stops the scan.</param>
    public List<GalleryEntry> Load(CancellationToken cancellation = default)
    {
        var entries = new List<GalleryEntry>();
        if (!Directory.Exists(_folder))
        {
            return entries;
        }

        // One directory scan per kind of file instead of a stat per image; the sidecars are read on a few threads.
        var pngs = new HashSet<string>(Directory.EnumerateFiles(_folder, "*.png", SearchOption.AllDirectories), StringComparer.Ordinal);
        var sidecars = Directory.EnumerateFiles(_folder, "*.json", SearchOption.AllDirectories).Where(sidecar => pngs.Contains(Path.ChangeExtension(sidecar, ".png"))).ToList();
        var options = new ParallelOptions { MaxDegreeOfParallelism = 4, CancellationToken = cancellation };
        Parallel.ForEach(sidecars, options, sidecar =>
        {
            try
            {
                var entry = JsonSerializer.Deserialize(File.ReadAllText(sidecar), GalleryJsonContext.Default.GalleryEntry);
                if (entry is not null)
                {
                    // The location on disk wins over what the sidecar says, so moved folders keep working.
                    string relative = Path.GetRelativePath(_folder, Path.ChangeExtension(sidecar, ".png")).Replace(Path.DirectorySeparatorChar, '/');
                    lock (entries)
                    {
                        entries.Add(entry with { File = relative });
                    }
                }
            }
            catch (Exception error) when (error is JsonException or IOException or UnauthorizedAccessException)
            {
                // Not one of ours or damaged: left alone.
            }
        });

        entries.Sort((a, b) => b.Created.CompareTo(a.Created));
        return entries;
    }

    /// <summary>Reads an entry's PNG.</summary>
    public byte[] ReadPng(GalleryEntry entry) => File.ReadAllBytes(PathOf(entry));

    /// <summary>Removes an entry's PNG and sidecar (and the day folder when it became empty).</summary>
    public void Delete(GalleryEntry entry)
    {
        string path = PathOf(entry);
        try
        {
            File.Delete(path);
            File.Delete(Path.ChangeExtension(path, ".json"));
            string directory = Path.GetDirectoryName(path)!;
            if (entry.Video.Length > 0)
            {
                File.Delete(PathOfVideo(entry));
            }

            if (Directory.Exists(directory) && !Directory.EnumerateFileSystemEntries(directory).Any())
            {
                Directory.Delete(directory);
            }
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            // Left on disk; it shows up again at the next start.
        }
    }
}
