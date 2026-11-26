using System.Text.Json;

namespace Sharplight.Hub;

/// <summary>
/// Recognizes and plans the download of PyTorch checkpoints (<see cref="ModelFormat.PyTorch"/>): Hugging Face checkpoints with
/// <c>pytorch_model*.bin</c> weights (configuration, tokenizer, index and shards), checkpoint folders in their authors' layout (the
/// <c>.pth</c> / <c>.pt</c> / <c>.ckpt</c> files at the root with their configuration, e.g. SeedVR2's DiT, VAE and text embeddings) and
/// single <c>.ckpt</c> / <c>.pth</c> files. Training state (optimizer, scheduler, random state, <c>checkpoint-N/</c> folders), Python
/// wheels, images and the <c>.bin</c> files of other runtimes (OpenVINO, Core ML, CTranslate2, whisper.cpp) are never taken.
/// </summary>
/// <remarks>
/// A folder with several large checkpoints offers each as a variant: a root weight file at least half as large as the largest is an
/// alternative (<c>seedvr2_ema_7b.pth</c> or <c>seedvr2_ema_7b_sharp.pth</c>), the smaller ones are shared by every variant
/// (<c>ema_vae.pth</c>, <c>pos_emb.pt</c>). The variant with the shortest name is the default.
/// </remarks>
internal static class PyTorchSelection
{
    private const string ConfigFile = "config.json";
    private const string BinIndexSuffix = ".bin.index.json";
    private const string WeightMapProperty = "weight_map";

    private static readonly string[] s_checkpointExtensions = [".pth", ".pt", ".ckpt"];
    private static readonly string[] s_binPrefixes = ["pytorch_model", "diffusion_pytorch_model"];
    private static readonly string[] s_configExtensions = [".json", ".yaml", ".yml", ".jinja"];
    private static readonly string[] s_tokenizerExtensions = [".txt", ".model", ".tiktoken"];
    private static readonly string[] s_trainingStateNames = ["optimizer", "scheduler", "scaler", "rng_state", "training_args", "training_state", "trainer_state"];
    private static readonly string[] s_foreignRuntimeDirectories = [".mlmodelc", ".mlpackage", "openvino"];

    /// <summary>Whether the repository has PyTorch model weights (anywhere in it).</summary>
    /// <param name="paths">Repository paths.</param>
    public static bool HasWeights(IReadOnlyList<string> paths) => paths.Any(path => IsWeight(path, paths));

    /// <summary>Whether <paramref name="path"/> is a PyTorch model weight file of a repository with <paramref name="paths"/>.</summary>
    /// <param name="path">A repository path.</param>
    /// <param name="paths">All paths of the repository (an index next to a <c>.bin</c> file marks it as a shard).</param>
    public static bool IsWeight(string path, IReadOnlyList<string> paths)
    {
        if (IsTrainingState(path) || IsForeignRuntime(path))
        {
            return false;
        }

        string name = RepoPath.Name(path);
        if (s_checkpointExtensions.Any(extension => name.EndsWith(extension, StringComparison.OrdinalIgnoreCase)))
        {
            return true;
        }

        if (!name.EndsWith(".bin", StringComparison.OrdinalIgnoreCase) || RepoLayouts.IsWhisperCpp(path))
        {
            return false;
        }

        if (s_binPrefixes.Any(prefix => name.StartsWith(prefix, StringComparison.OrdinalIgnoreCase)))
        {
            return true;
        }

        // Other names count as shards when an index sits beside them.
        string directory = RepoPath.Directory(path);
        return paths.Any(other => RepoPath.Directory(other) == directory && other.EndsWith(BinIndexSuffix, StringComparison.OrdinalIgnoreCase));
    }

    /// <summary>The <c>pytorch_model.bin.index.json</c> of a Hugging Face checkpoint at the root, or null.</summary>
    /// <param name="paths">Repository paths.</param>
    public static string? IndexPath(IReadOnlyList<string> paths)
        => paths.Where(RepoPath.IsRoot).Where(path => path.EndsWith(BinIndexSuffix, StringComparison.OrdinalIgnoreCase)).Order(StringComparer.Ordinal).FirstOrDefault();

    /// <summary>The variant tags (file names without extension of the alternative checkpoints), the default first; empty for a single checkpoint.</summary>
    /// <param name="files">The repository's files (with sizes).</param>
    public static IReadOnlyList<string> Variants(IReadOnlyList<HubFile> files)
    {
        if (IsHuggingFaceCheckpoint(files))
        {
            return [];
        }

        var alternatives = Alternatives(files);
        return alternatives.Count > 1 ? [.. alternatives.Select(file => Stem(file.Name))] : [];
    }

    /// <summary>Chooses the files of the variant <paramref name="tag"/> (null or <see cref="ModelFileSelector.DefaultTag"/>: the default).</summary>
    /// <param name="repository">The repository.</param>
    /// <param name="tag">A tag from <see cref="Variants"/>, a weight file name, or null.</param>
    /// <param name="indexJson">The text of <see cref="IndexPath"/> for a sharded Hugging Face checkpoint, or null.</param>
    /// <exception cref="HubException">The repository has no PyTorch weights, the tag matches no variant, or the index is not valid.</exception>
    public static ModelSelection Select(HubRepository repository, string? tag, string? indexJson)
    {
        bool defaultTag = tag is null || string.Equals(tag, ModelFileSelector.DefaultTag, StringComparison.OrdinalIgnoreCase);
        var paths = repository.Files.Select(file => file.Path).ToList();
        if (IsHuggingFaceCheckpoint(repository.Files))
        {
            if (!defaultTag)
            {
                throw new HubException($"{repository.Id} is a single PyTorch checkpoint without variants; drop ':{tag}'.");
            }

            var root = repository.Files.Where(file => RepoPath.IsRoot(file.Path)).ToList();
            var weights = indexJson is null
                ? root.Where(file => IsWeight(file.Path, paths))
                : IndexedShards(repository, indexJson);
            var metadata = root.Where(file => IsConfiguration(file.Name) || IsTokenizer(file.Name));
            return Selection("", [.. metadata, .. weights], "");
        }

        var alternatives = Alternatives(repository.Files);
        if (alternatives.Count == 0)
        {
            throw new HubException($"{repository.Id} has no PyTorch model files.");
        }

        HubFile chosen;
        string label;
        if (defaultTag)
        {
            chosen = alternatives[0];
            label = alternatives.Count > 1 ? Stem(chosen.Name) : "";
        }
        else
        {
            chosen = alternatives.FirstOrDefault(file => string.Equals(Stem(file.Name), tag, StringComparison.OrdinalIgnoreCase)
                    || string.Equals(file.Name, tag, StringComparison.OrdinalIgnoreCase) || string.Equals(file.Path, tag, StringComparison.OrdinalIgnoreCase))
                ?? throw new HubException(alternatives.Count == 1
                    ? $"{repository.Id} has a single PyTorch checkpoint; drop ':{tag}'."
                    : $"{repository.Id} has no variant '{tag}'. Available: {string.Join(", ", alternatives.Select(file => Stem(file.Name)))}.");
            label = alternatives.Count > 1 ? Stem(chosen.Name) : "";
        }

        string directory = chosen.Directory;
        var inFolder = repository.Files.Where(file => file.Directory == directory).ToList();
        var shared = inFolder.Where(file => IsWeight(file.Path, paths) && !alternatives.Contains(file));
        var configuration = inFolder.Where(file => IsConfiguration(file.Name));
        var selected = new List<HubFile> { chosen };
        selected.AddRange(shared);
        selected.AddRange(configuration);

        // A lone checkpoint file without configuration is opened as that file; a folder layout as its folder.
        string primary = selected.Count == 1 ? chosen.Path : directory;
        return Selection(label, selected, primary);
    }

    /// <summary>A Hugging Face checkpoint: <c>config.json</c> and <c>pytorch_model*.bin</c> weights at the root.</summary>
    private static bool IsHuggingFaceCheckpoint(IReadOnlyList<HubFile> files)
    {
        var root = files.Where(file => RepoPath.IsRoot(file.Path)).ToList();
        return root.Any(file => file.Path == ConfigFile)
            && root.Any(file => file.Name.EndsWith(".bin", StringComparison.OrdinalIgnoreCase) && s_binPrefixes.Any(prefix => file.Name.StartsWith(prefix, StringComparison.OrdinalIgnoreCase)));
    }

    /// <summary>
    /// The checkpoints that are alternatives of each other, the default first: the weight files of the folder that holds the largest one
    /// (the root when it has weights) that are at least half its size.
    /// </summary>
    private static List<HubFile> Alternatives(IReadOnlyList<HubFile> files)
    {
        var paths = files.Select(file => file.Path).ToList();
        var weights = files.Where(file => IsWeight(file.Path, paths)).ToList();
        var root = weights.Where(file => RepoPath.IsRoot(file.Path)).ToList();
        var candidates = root.Count > 0 ? root : weights;
        if (candidates.Count == 0)
        {
            return [];
        }

        var largest = candidates.MaxBy(file => file.Size)!;
        return [.. candidates.Where(file => file.Directory == largest.Directory && file.Size * 2 >= largest.Size)
            .OrderBy(file => file.Name.Length).ThenBy(file => file.Name, StringComparer.Ordinal)];
    }

    private static ModelSelection Selection(string label, IEnumerable<HubFile> files, string primary)
        => new(ModelFormat.PyTorch, label, [.. files.DistinctBy(file => file.Path).OrderBy(file => file.Path, StringComparer.Ordinal)], primary, null);

    private static IEnumerable<HubFile> IndexedShards(HubRepository repository, string indexJson)
    {
        HashSet<string> shardNames;
        try
        {
            using var document = JsonDocument.Parse(indexJson);
            shardNames = document.RootElement.GetProperty(WeightMapProperty).EnumerateObject()
                .Select(entry => entry.Value.GetString() ?? "")
                .ToHashSet(StringComparer.Ordinal);
        }
        catch (Exception error) when (error is JsonException or KeyNotFoundException or InvalidOperationException)
        {
            throw new HubException($"{repository.Id}: the PyTorch checkpoint index is not valid ({error.Message}).", inner: error);
        }

        var shards = new List<HubFile>();
        foreach (var name in shardNames.Order(StringComparer.Ordinal))
        {
            shards.Add(repository.Files.FirstOrDefault(file => file.Path == name)
                ?? throw new HubException($"{repository.Id}: the PyTorch checkpoint index lists '{name}', which is not in the repository."));
        }

        return shards;
    }

    /// <summary>
    /// Optimizer, scheduler, gradient-scaler, random-number and trainer state (<c>optimizer.pt</c>, <c>rng_state_0.pth</c>,
    /// <c>training_args.bin</c>), and anything in a <c>checkpoint-N/</c> or <c>global_stepN/</c> training folder.
    /// </summary>
    private static bool IsTrainingState(string path)
    {
        string stem = Stem(RepoPath.Name(path));
        return s_trainingStateNames.Any(state => stem.Equals(state, StringComparison.OrdinalIgnoreCase) || stem.StartsWith(state + "_", StringComparison.OrdinalIgnoreCase))
            || path.Split('/')[..^1].Any(segment => segment.StartsWith("checkpoint-", StringComparison.OrdinalIgnoreCase) || segment.StartsWith("global_step", StringComparison.OrdinalIgnoreCase));
    }

    /// <summary>A file of another runtime's model: Core ML bundles and OpenVINO IR.</summary>
    private static bool IsForeignRuntime(string path)
        => path.Split('/')[..^1].Any(segment => s_foreignRuntimeDirectories.Any(suffix => segment.EndsWith(suffix, StringComparison.OrdinalIgnoreCase)))
            || RepoPath.Name(path).StartsWith("openvino_", StringComparison.OrdinalIgnoreCase);

    private static bool IsConfiguration(string name)
        => s_configExtensions.Any(extension => name.EndsWith(extension, StringComparison.OrdinalIgnoreCase)) && !IsTrainingState(name);

    private static bool IsTokenizer(string name)
        => s_tokenizerExtensions.Any(extension => name.EndsWith(extension, StringComparison.OrdinalIgnoreCase))
            && !name.Equals("requirements.txt", StringComparison.OrdinalIgnoreCase) && !name.StartsWith("README", StringComparison.OrdinalIgnoreCase);

    private static string Stem(string name) => name.LastIndexOf('.') is var dot and > 0 ? name[..dot] : name;
}
