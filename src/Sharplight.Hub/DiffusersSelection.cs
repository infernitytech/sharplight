using System.Text.RegularExpressions;

namespace Sharplight.Hub;

/// <summary>
/// Plans the download of a Diffusers pipeline: <c>model_index.json</c>, every used component's configuration, tokenizer and scheduler
/// files and one weight variant per component (safetensors only; a half-precision or full-precision copy; never the <c>.bin</c>, ONNX
/// or Flax duplicates). The safety checker's weights are not needed and are left out.
/// </summary>
internal static partial class DiffusersSelection
{
    /// <summary>Tag of the full-precision variant.</summary>
    public const string FullTag = "full";

    private static readonly string[] s_textExtensions = [".json", ".txt", ".model", ".jinja", ".tiktoken"];
    private static readonly string[] s_skippedWeights = ["safety_checker", "feature_extractor"];
    private static readonly string[] s_precisions = ["fp16", "bf16"];

    [GeneratedRegex(@"-\d{5}-of-\d{5}", RegexOptions.CultureInvariant)]
    private static partial Regex ShardPattern();

    /// <summary>The variants the repository offers, preferred (smallest) first; empty when there is only one kind of weights.</summary>
    /// <param name="files">The repository's files.</param>
    /// <param name="pipeline">The parsed index, or null to use every folder.</param>
    public static IReadOnlyList<string> Variants(IReadOnlyList<HubFile> files, DiffusersPipeline? pipeline)
    {
        var weights = ComponentWeights(files, pipeline).SelectMany(group => group.Value).ToList();
        var variants = new List<string>();
        foreach (var precision in s_precisions)
        {
            if (weights.Any(file => Precision(file.Name) == precision))
            {
                variants.Add(precision);
            }
        }

        if (weights.Any(file => Precision(file.Name).Length == 0))
        {
            variants.Add(FullTag);
        }

        return variants.Count > 1 ? variants : [];
    }

    /// <summary>Chooses the files of the variant <paramref name="tag"/> (null: the first variant).</summary>
    /// <param name="repository">The repository.</param>
    /// <param name="tag">A tag from <see cref="Variants"/>, or null.</param>
    /// <param name="indexJson">The text of <c>model_index.json</c>, or null to use every folder.</param>
    public static ModelSelection Select(HubRepository repository, string? tag, string? indexJson)
    {
        var pipeline = indexJson is null ? null : DiffusersPipeline.Parse(indexJson);
        var variants = Variants(repository.Files, pipeline);
        string label = "";
        string? wanted = null;
        if (tag is not null && !string.Equals(tag, ModelFileSelector.DefaultTag, StringComparison.OrdinalIgnoreCase))
        {
            label = variants.FirstOrDefault(variant => string.Equals(variant, tag, StringComparison.OrdinalIgnoreCase))
                ?? throw new HubException(variants.Count == 0
                    ? $"{repository.Id} has a single weight variant; drop ':{tag}'."
                    : $"{repository.Id} has no variant '{tag}'. Available: {string.Join(", ", variants)}.");
            wanted = label == FullTag ? "" : label;
        }
        else if (variants.Count > 0)
        {
            label = variants[0];
            wanted = label == FullTag ? "" : label;
        }

        var selected = new List<HubFile>();
        selected.AddRange(repository.Files.Where(file => RepoPath.IsRoot(file.Path) && file.Path.EndsWith(".json", StringComparison.OrdinalIgnoreCase)));
        var folders = FolderSet(repository.Files, pipeline);
        var weightFiles = new HashSet<string>(StringComparer.Ordinal);
        bool chosenFp16Index;
        foreach (var (component, weights) in ComponentWeights(repository.Files, pipeline))
        {
            var chosen = ChooseWeights(weights, wanted);
            selected.AddRange(chosen);
            weightFiles.UnionWith(weights.Select(file => file.Path));
            chosenFp16Index = chosen.Any(file => Precision(file.Name).Length > 0);
            string chosenPrecision = chosen.Count == 0 ? "" : Precision(chosen[0].Name);
            selected.AddRange(repository.Files.Where(file => RepoPath.Directory(file.Path) == component && IsIndex(file.Name)
                && (chosenFp16Index ? Precision(file.Name) == chosenPrecision : Precision(file.Name).Length == 0)));
        }

        foreach (var file in repository.Files)
        {
            string directory = RepoPath.Directory(file.Path);
            if (directory.Length > 0 && !directory.Contains('/') && folders.Contains(directory) && !weightFiles.Contains(file.Path)
                && !IsIndex(file.Name) && s_textExtensions.Any(extension => file.Path.EndsWith(extension, StringComparison.OrdinalIgnoreCase)))
            {
                selected.Add(file);
            }
        }

        return new ModelSelection(ModelFormat.Diffusers, label, [.. selected.DistinctBy(file => file.Path).OrderBy(file => file.Path, StringComparer.Ordinal)], "", null);
    }

    private static bool IsIndex(string name) => name.EndsWith(".json", StringComparison.OrdinalIgnoreCase) && name.Contains(".index", StringComparison.OrdinalIgnoreCase);

    /// <summary>The precision a weight file name declares (<c>fp16</c>, <c>bf16</c>), or empty for full precision.</summary>
    private static string Precision(string name)
        => s_precisions.FirstOrDefault(precision => name.Contains("." + precision, StringComparison.OrdinalIgnoreCase)) ?? "";

    private static HashSet<string> FolderSet(IReadOnlyList<HubFile> files, DiffusersPipeline? pipeline)
    {
        if (pipeline is { Components.Count: > 0 })
        {
            return new HashSet<string>(pipeline.Components, StringComparer.Ordinal);
        }

        return files.Where(file => !RepoPath.IsRoot(file.Path)).Select(file => file.Path[..file.Path.IndexOf('/')]).ToHashSet(StringComparer.Ordinal);
    }

    /// <summary>The safetensors files of each component folder that needs weights.</summary>
    private static Dictionary<string, List<HubFile>> ComponentWeights(IReadOnlyList<HubFile> files, DiffusersPipeline? pipeline)
    {
        var folders = FolderSet(files, pipeline);
        var result = new Dictionary<string, List<HubFile>>(StringComparer.Ordinal);
        foreach (var folder in folders.Order(StringComparer.Ordinal))
        {
            var inFolder = files.Where(file => RepoPath.Directory(file.Path) == folder).ToList();
            var safetensors = inFolder.Where(file => file.Path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase)).ToList();
            if (safetensors.Count > 0 && !s_skippedWeights.Contains(folder, StringComparer.Ordinal))
            {
                result[folder] = safetensors;
            }
            else if (safetensors.Count == 0 && !s_skippedWeights.Contains(folder, StringComparer.Ordinal)
                && inFolder.Any(file => file.Name.EndsWith(".bin", StringComparison.OrdinalIgnoreCase) || file.Name.EndsWith(".ckpt", StringComparison.OrdinalIgnoreCase)
                    || file.Name.EndsWith(".pth", StringComparison.OrdinalIgnoreCase) || file.Name.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase)))
            {
                throw new HubException($"The '{folder}' component ships no .safetensors weights (only .bin, .ckpt or ONNX copies), which Sharplight cannot read.");
            }
        }

        return result;
    }

    /// <summary>One precision's weights of a component: all shards of a split checkpoint, else one file (not an EMA copy).</summary>
    private static List<HubFile> ChooseWeights(IReadOnlyList<HubFile> weights, string? wanted)
    {
        var order = new List<string>();
        if (wanted is not null)
        {
            order.Add(wanted);
        }

        order.AddRange(new[] { "", "fp16", "bf16" }.Where(precision => precision != wanted));
        foreach (var precision in order)
        {
            var group = weights.Where(file => Precision(file.Name) == precision).OrderBy(file => file.Path, StringComparer.Ordinal).ToList();
            if (group.Count == 0)
            {
                continue;
            }

            var shards = group.Where(file => ShardPattern().IsMatch(file.Name)).ToList();
            if (shards.Count > 0)
            {
                string prefix = file0(shards[0].Name);
                return [.. shards.Where(file => file0(file.Name) == prefix)];
            }

            return [group.OrderBy(file => file.Name.Contains("ema", StringComparison.OrdinalIgnoreCase) ? 1 : 0).ThenBy(file => file.Name.Length).First()];
        }

        return [];

        static string file0(string name) => name[..ShardPattern().Match(name).Index];
    }
}
