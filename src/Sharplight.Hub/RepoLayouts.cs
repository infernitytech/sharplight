namespace Sharplight.Hub;

/// <summary>
/// Recognizes repositories whose model is more than one file: Diffusers pipelines (a folder per component) and whisper.cpp ggml
/// files (one file per size and quantization), so they are listed and shown as one model.
/// </summary>
public static class RepoLayouts
{
    /// <summary>The file that marks a Diffusers pipeline.</summary>
    public const string DiffusersIndex = "model_index.json";

    private static readonly string[] s_weightExtensions = [".safetensors", ".bin", ".onnx", ".ckpt", ".pt", ".pth"];

    /// <summary>True for a whisper.cpp model file at the repository root (<c>ggml-small.en-q5_1.bin</c>).</summary>
    /// <param name="path">A repository path.</param>
    public static bool IsWhisperCpp(string path)
        => RepoPath.IsRoot(path) && path.StartsWith("ggml-", StringComparison.OrdinalIgnoreCase) && path.EndsWith(".bin", StringComparison.OrdinalIgnoreCase);

    /// <summary>The components of a Diffusers pipeline: one per subfolder that holds weights, largest first.</summary>
    /// <param name="files">The repository's files.</param>
    public static IReadOnlyList<HubComponent> DiffusersComponents(IEnumerable<HubFile> files)
    {
        var components = new List<HubComponent>();
        foreach (var group in files.Where(file => !RepoPath.IsRoot(file.Path) && IsWeight(file.Path)).GroupBy(file => file.Path[..file.Path.IndexOf('/')], StringComparer.Ordinal))
        {
            // A component often ships .safetensors and .bin (or fp16 copies) of the same weights: count the preferred copy once.
            var weights = group.ToList();
            var safetensors = weights.Where(file => file.Path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase)).ToList();
            var chosen = safetensors.Count > 0 ? safetensors : weights;
            var plain = chosen.Where(file => !RepoPath.Name(file.Path).Contains(".fp16.", StringComparison.OrdinalIgnoreCase)
                && !RepoPath.Name(file.Path).Contains(".bf16.", StringComparison.OrdinalIgnoreCase)).ToList();
            if (plain.Count > 0)
            {
                chosen = plain;
            }

            components.Add(new HubComponent(group.Key, chosen.Sum(file => file.Size), chosen.Count));
        }

        return [.. components.OrderByDescending(component => component.Bytes).ThenBy(component => component.Name, StringComparer.Ordinal)];
    }

    /// <summary>The whisper.cpp model files of a repository as components named by their size and quantization (<c>small.en-q5_1</c>).</summary>
    /// <param name="files">The repository's files.</param>
    public static IReadOnlyList<HubComponent> WhisperModels(IEnumerable<HubFile> files)
    {
        return [.. files.Where(file => IsWhisperCpp(file.Path)).OrderBy(file => file.Size).ThenBy(file => file.Path, StringComparer.Ordinal)
            .Select(file => new HubComponent(file.Path["ggml-".Length..^".bin".Length], file.Size, 1))];
    }

    /// <summary>
    /// The model format of a repository Sharplight cannot read, named for people (<c>NeMo archives (.nemo)</c>), or null when it has
    /// none of the formats listed here. Used to say why a repository without supported model files cannot be pulled.
    /// </summary>
    /// <param name="paths">Repository paths.</param>
    public static string? UnsupportedFormat(IEnumerable<string> paths)
    {
        var list = paths.ToList();
        bool Any(Func<string, bool> predicate) => list.Any(predicate);
        bool Ends(string path, string extension) => path.EndsWith(extension, StringComparison.OrdinalIgnoreCase);
        bool Named(string path, string name) => string.Equals(RepoPath.Name(path), name, StringComparison.OrdinalIgnoreCase);
        if (Any(path => Ends(path, ".nemo")))
        {
            return "NeMo archives (.nemo)";
        }

        if (Any(path => RepoPath.IsRoot(path) && Named(path, "model.bin")) && Any(path => RepoPath.IsRoot(path) && RepoPath.Name(path).StartsWith("vocabulary.", StringComparison.OrdinalIgnoreCase)))
        {
            return "CTranslate2 weights (model.bin)";
        }

        if (Any(path => path.Contains(".mlmodelc/", StringComparison.OrdinalIgnoreCase) || path.Contains(".mlpackage/", StringComparison.OrdinalIgnoreCase) || Ends(path, ".mlmodel")))
        {
            return "Core ML models";
        }

        if (Any(path => Ends(path, ".xml")) && Any(path => RepoPath.Name(path).StartsWith("openvino", StringComparison.OrdinalIgnoreCase) || path.Contains("openvino/", StringComparison.OrdinalIgnoreCase)))
        {
            return "OpenVINO models";
        }

        if (Any(path => Ends(path, ".tflite")))
        {
            return "TensorFlow Lite models (.tflite)";
        }

        if (Any(path => Named(path, "saved_model.pb") || Ends(path, ".h5") || Ends(path, ".keras")))
        {
            return "TensorFlow / Keras weights";
        }

        if (Any(path => Ends(path, ".msgpack")))
        {
            return "Flax weights (.msgpack)";
        }

        if (Any(path => Ends(path, ".pdiparams") || Ends(path, ".pdparams")))
        {
            return "PaddlePaddle weights";
        }

        if (Any(path => Ends(path, ".pkl") || Ends(path, ".pickle")))
        {
            return "pickled Python objects (.pkl)";
        }

        if (Any(path => Ends(path, ".npz")))
        {
            return "NumPy archives (.npz)";
        }

        return Any(path => Ends(path, ".engine") || Ends(path, ".plan")) ? "TensorRT engines" : null;
    }

    private static bool IsWeight(string path) => s_weightExtensions.Any(extension => path.EndsWith(extension, StringComparison.OrdinalIgnoreCase));
}
