using System.Text.RegularExpressions;

namespace Sharplight.Gguf;

/// <summary>
/// One logical model that may be split over several GGUF shards (<c>name-00001-of-00003.gguf</c>).
/// Metadata comes from the first shard; tensor lists are merged. Disposing unmaps every shard.
/// </summary>
public sealed partial class GgufModelFiles : IDisposable
{
    private readonly List<GgufFile> _files;
    private readonly List<GgufTensorInfo> _tensors = [];
    private readonly Dictionary<string, GgufTensorInfo> _byName = new(StringComparer.Ordinal);

    [GeneratedRegex(@"^(?<prefix>.*)-(?<idx>\d{5})-of-(?<n>\d{5})\.gguf$", RegexOptions.IgnoreCase)]
    private static partial Regex ShardRegex();

    private GgufModelFiles(List<GgufFile> files)
    {
        _files = files;
        foreach (var file in files)
        {
            foreach (var tensor in file.Tensors)
            {
                _tensors.Add(tensor);
                _byName[tensor.Name] = tensor;
            }
        }
    }

    /// <summary>The opened shards in order (one entry for an unsplit model).</summary>
    public IReadOnlyList<GgufFile> Files => _files;

    /// <summary>The first shard, which holds the model's metadata.</summary>
    public GgufFile First => _files[0];

    /// <summary>The metadata of the first shard.</summary>
    public IReadOnlyDictionary<string, object> Metadata => First.Metadata;

    /// <summary>The tensors of all shards, shard by shard in file order.</summary>
    public IReadOnlyList<GgufTensorInfo> Tensors => _tensors;

    internal string? Architecture => First.Architecture;

    internal GgufTensorInfo? FindTensor(string name) => _byName.GetValueOrDefault(name);

    /// <summary>Opens a single file, or all shards when <paramref name="path"/> names a split shard.</summary>
    /// <param name="path">A <c>.gguf</c> file; for a split model the first shard (<c>-00001-of-N</c>).</param>
    /// <exception cref="ArgumentException"><paramref name="path"/> names a shard other than the first.</exception>
    /// <exception cref="InvalidDataException">A file is not a well-formed GGUF file.</exception>
    public static GgufModelFiles Open(string path)
    {
        var files = new List<GgufFile>();
        try
        {
            var shardMatch = ShardRegex().Match(System.IO.Path.GetFileName(path));
            var first = GgufFile.Open(path);
            files.Add(first);
            if (shardMatch.Success)
            {
                OpenRemainingShards(path, shardMatch, first, files);
            }

            return new GgufModelFiles(files);
        }
        catch
        {
            foreach (var file in files)
            {
                file.Dispose();
            }

            throw;
        }
    }

    private static void OpenRemainingShards(string path, Match shardMatch, GgufFile first, List<GgufFile> files)
    {
        int index = int.Parse(shardMatch.Groups["idx"].Value);
        int count = int.Parse(shardMatch.Groups["n"].Value);
        if (first.TryGet<int>("split.count", out var declaredCount) && declaredCount > 0)
        {
            count = declaredCount;
        }

        if (index != 1)
        {
            throw new ArgumentException("Pass the first shard (-00001-of-N) of a split model.", nameof(path));
        }

        string directory = System.IO.Path.GetDirectoryName(path) ?? ".";
        string prefix = shardMatch.Groups["prefix"].Value;
        int width = shardMatch.Groups["idx"].Length;
        for (int i = 2; i <= count; i++)
        {
            string shardIndex = i.ToString().PadLeft(width, '0');
            string shardCount = count.ToString().PadLeft(width, '0');
            files.Add(GgufFile.Open(System.IO.Path.Combine(directory, $"{prefix}-{shardIndex}-of-{shardCount}.gguf")));
        }
    }

    /// <summary>Unmaps all shards.</summary>
    public void Dispose()
    {
        foreach (var file in _files)
        {
            file.Dispose();
        }
    }
}
