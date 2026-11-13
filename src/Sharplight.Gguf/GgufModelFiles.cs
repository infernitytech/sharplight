using System.Text.RegularExpressions;

namespace Sharplight.Gguf;

/// <summary>
/// One logical model that may be split over several GGUF shards (<c>name-00001-of-00003.gguf</c>).
/// Metadata comes from the first shard; tensor lists are merged.
/// </summary>
public sealed partial class GgufModelFiles : IDisposable
{
    readonly List<GgufFile> _files;
    readonly List<GgufTensorInfo> _tensors = [];
    readonly Dictionary<string, GgufTensorInfo> _byName = new(StringComparer.Ordinal);

    [GeneratedRegex(@"^(?<prefix>.*)-(?<idx>\d{5})-of-(?<n>\d{5})\.gguf$", RegexOptions.IgnoreCase)]
    private static partial Regex ShardRegex();

    GgufModelFiles(List<GgufFile> files)
    {
        _files = files;
        foreach (var f in files)
            foreach (var t in f.Tensors) { _tensors.Add(t); _byName[t.Name] = t; }
    }

    public IReadOnlyList<GgufFile> Files => _files;
    public GgufFile First => _files[0];
    public IReadOnlyDictionary<string, object> Metadata => First.Metadata;
    public IReadOnlyList<GgufTensorInfo> Tensors => _tensors;
    public string? Architecture => First.Architecture;
    public GgufTensorInfo? FindTensor(string name) => _byName.GetValueOrDefault(name);
    public bool TryGetTensor(string name, out GgufTensorInfo info) => _byName.TryGetValue(name, out info!);

    /// <summary>Opens a single file, or all shards when <paramref name="path"/> names a split shard.</summary>
    public static GgufModelFiles Open(string path)
    {
        var files = new List<GgufFile>();
        try
        {
            var m = ShardRegex().Match(System.IO.Path.GetFileName(path));
            var first = GgufFile.Open(path);
            files.Add(first);
            if (m.Success)
            {
                int index = int.Parse(m.Groups["idx"].Value);
                int count = int.Parse(m.Groups["n"].Value);
                if (first.TryGet<int>("split.count", out var sc) && sc > 0) count = sc;
                if (index != 1) throw new ArgumentException("Pass the first shard (-00001-of-N) of a split model.", nameof(path));
                string dir = System.IO.Path.GetDirectoryName(path) ?? ".";
                string prefix = m.Groups["prefix"].Value;
                int width = m.Groups["idx"].Length;
                for (int i = 2; i <= count; i++)
                {
                    string name = $"{prefix}-{i.ToString().PadLeft(width, '0')}-of-{count.ToString().PadLeft(width, '0')}.gguf";
                    files.Add(GgufFile.Open(System.IO.Path.Combine(dir, name)));
                }
            }
            return new GgufModelFiles(files);
        }
        catch
        {
            foreach (var f in files) f.Dispose();
            throw;
        }
    }

    public void Dispose() { foreach (var f in _files) f.Dispose(); }
}
