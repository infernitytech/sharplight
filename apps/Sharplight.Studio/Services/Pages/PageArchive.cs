namespace Sharplight.Studio.Services.Pages;

/// <summary>
/// The persisted runs of one inference page: the history (<c>history.json</c>) and the output files of every run
/// (<c>outputs/&lt;run id&gt;/</c>). Both the history list and the outputs gallery show it.
/// </summary>
internal sealed class PageArchive
{
    /// <summary>The runs kept per page; older ones (and their files) are removed.</summary>
    public const int Capacity = 200;

    private readonly string _directory;
    private readonly List<RunRecord> _runs;

    /// <param name="storage">The page storage.</param>
    /// <param name="pageId">The page's id.</param>
    public PageArchive(IPageStorage storage, string pageId)
    {
        _directory = storage.DirectoryFor(pageId);
        _runs = PageJson.Read(HistoryFile, StudioJsonContext.Default.RunHistoryFile).Runs;
    }

    /// <summary>Raised after a run was added or removed.</summary>
    public event EventHandler? Changed;

    /// <summary>The runs, newest first.</summary>
    public IReadOnlyList<RunRecord> Runs => _runs;

    private string HistoryFile => Path.Combine(_directory, "history.json");

    /// <summary>The directory of a run's output files.</summary>
    /// <param name="record">The run.</param>
    public string DirectoryOf(RunRecord record) => Path.Combine(_directory, "outputs", record.Id);

    /// <summary>The path of one of a run's output files.</summary>
    /// <param name="record">The run.</param>
    /// <param name="file">The file's name (one of <see cref="RunRecord.Files"/>).</param>
    public string PathOf(RunRecord record, string file) => Path.Combine(DirectoryOf(record), Path.GetFileName(file));

    /// <summary>A new unique run id.</summary>
    public static string NewId() => DateTime.Now.ToString("yyyyMMdd-HHmmss-", System.Globalization.CultureInfo.InvariantCulture) + Guid.NewGuid().ToString("N")[..6];

    /// <summary>Adds a run at the top, writing its output files first.</summary>
    /// <param name="record">The run; its <see cref="RunRecord.Files"/> name the entries of <paramref name="files"/>.</param>
    /// <param name="files">The output files by name.</param>
    public void Add(RunRecord record, IReadOnlyDictionary<string, byte[]>? files = null)
    {
        if (files is { Count: > 0 })
        {
            string directory = DirectoryOf(record);
            Directory.CreateDirectory(directory);
            foreach (var (name, content) in files)
            {
                File.WriteAllBytes(Path.Combine(directory, Path.GetFileName(name)), content);
            }
        }

        _runs.Insert(0, record);
        while (_runs.Count > Capacity)
        {
            DeleteFiles(_runs[^1]);
            _runs.RemoveAt(_runs.Count - 1);
        }

        Save();
    }

    /// <summary>Removes a run and its output files.</summary>
    /// <param name="record">The run.</param>
    public void Delete(RunRecord record)
    {
        if (_runs.Remove(record))
        {
            DeleteFiles(record);
            Save();
        }
    }

    /// <summary>Removes every run and every output file.</summary>
    public void Clear()
    {
        foreach (var record in _runs)
        {
            DeleteFiles(record);
        }

        _runs.Clear();
        Save();
    }

    private void DeleteFiles(RunRecord record)
    {
        try
        {
            if (Directory.Exists(DirectoryOf(record)))
            {
                Directory.Delete(DirectoryOf(record), recursive: true);
            }
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            // The files stay; the record is gone.
        }
    }

    private void Save()
    {
        PageJson.Write(HistoryFile, new RunHistoryFile { Runs = _runs }, StudioJsonContext.Default.RunHistoryFile);
        Changed?.Invoke(this, EventArgs.Empty);
    }
}
