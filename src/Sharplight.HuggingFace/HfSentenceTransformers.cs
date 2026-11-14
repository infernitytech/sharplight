using System.Text.Json;
using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>
/// Translates sentence-transformers metadata stored next to a checkpoint (<c>modules.json</c>, the Pooling module's config,
/// <c>config_sentence_transformers.json</c>, <c>sentence_bert_config.json</c>) and cross-encoder hints in <c>config.json</c> into the
/// canonical <c>embedding.*</c> / <c>classifier.*</c> keys.
/// </summary>
internal static class HfSentenceTransformers
{
    /// <summary>Adds the canonical embedding / classifier keys found in the sentence-transformers files of the model directory to <paramref name="md"/>.</summary>
    /// <param name="md">The canonical metadata being built.</param>
    /// <param name="dir">The model directory.</param>
    /// <param name="config">The checkpoint's <c>config.json</c>.</param>
    public static void AddTo(IDictionary<string, object> md, string dir, HfConfig config)
    {
        AddModules(md, dir);
        AddPrompts(md, dir);
        AddMaxSequenceLength(md, dir);
        AddCrossEncoder(md, config);
    }

    /// <summary>Pooling / normalization from <c>modules.json</c>.</summary>
    private static void AddModules(IDictionary<string, object> md, string dir)
    {
        string modules = Path.Combine(dir, "modules.json");
        if (!File.Exists(modules))
        {
            return;
        }

        using var doc = JsonDocument.Parse(File.ReadAllBytes(modules));
        foreach (var module in doc.RootElement.EnumerateArray())
        {
            string type = module.TryGetProperty("type", out var typeElement) ? typeElement.GetString() ?? "" : "";
            string path = module.TryGetProperty("path", out var pathElement) ? pathElement.GetString() ?? "" : "";
            if (type.EndsWith(".Pooling", StringComparison.Ordinal) && Pooling(Path.Combine(dir, path, "config.json")) is { } pooling)
            {
                md[CanonicalKeys.EmbeddingPooling] = pooling;
            }
            else if (type.EndsWith(".Normalize", StringComparison.Ordinal))
            {
                md[CanonicalKeys.EmbeddingNormalize] = true;
            }
            else if (type.EndsWith(".Dense", StringComparison.Ordinal))
            {
                CheckDense(dir, path);
            }
        }

        if (!md.ContainsKey(CanonicalKeys.EmbeddingNormalize))
        {
            md[CanonicalKeys.EmbeddingNormalize] = false;
        }
    }

    /// <summary>
    /// The sentence-transformers Dense modules (in <c>modules.json</c> order) as canonical <c>embedding_projection.{i}.weight</c> / <c>.bias</c>
    /// weights, read from each module's <c>model.safetensors</c>; the opened files are added to <paramref name="owned"/>.
    /// </summary>
    /// <exception cref="NotSupportedException">A Dense module has a non-identity activation or no safetensors weights.</exception>
    public static IReadOnlyDictionary<string, WeightInfo> DenseWeights(string dir, List<IDisposable> owned)
    {
        var weights = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
        string modules = Path.Combine(dir, "modules.json");
        if (!File.Exists(modules))
        {
            return weights;
        }

        using var doc = JsonDocument.Parse(File.ReadAllBytes(modules));
        int index = 0;
        foreach (var module in doc.RootElement.EnumerateArray())
        {
            string type = module.TryGetProperty("type", out var typeElement) ? typeElement.GetString() ?? "" : "";
            if (!type.EndsWith(".Dense", StringComparison.Ordinal))
            {
                continue;
            }

            string path = Path.Combine(dir, module.GetProperty("path").GetString() ?? "");
            CheckDense(dir, path);
            var file = SafetensorsFile.Open(Path.Combine(path, "model.safetensors"));
            owned.Add(file);
            foreach (var (name, tensor) in file.Tensors)
            {
                string parameter = name.EndsWith("weight", StringComparison.Ordinal) ? "weight" : name.EndsWith("bias", StringComparison.Ordinal) ? "bias"
                    : throw new NotSupportedException($"{path}: unexpected Dense tensor '{name}'.");
                weights[$"embedding_projection.{index}.{parameter}"] = HfTransforms.Plain(tensor.ToWeightInfo(), $"embedding_projection.{index}.{parameter}");
            }

            index++;
        }

        return weights;
    }

    /// <summary>A Dense module must be linear (identity activation) and stored as safetensors.</summary>
    private static void CheckDense(string dir, string modulePath)
    {
        string module = Path.Combine(dir, modulePath);
        if (!File.Exists(Path.Combine(module, "model.safetensors")))
        {
            throw new NotSupportedException($"{module}: sentence-transformers Dense modules are only read from model.safetensors.");
        }

        string config = Path.Combine(module, "config.json");
        if (!File.Exists(config))
        {
            return;
        }

        using var doc = JsonDocument.Parse(File.ReadAllBytes(config));
        string activation = doc.RootElement.TryGetProperty("activation_function", out var a) ? a.GetString() ?? "" : "";
        if (activation.Length > 0 && !activation.EndsWith("Identity", StringComparison.Ordinal))
        {
            throw new NotSupportedException($"{module}: Dense activation '{activation}' is not supported (only Identity).");
        }
    }

    /// <summary>Named prompts from <c>config_sentence_transformers.json</c>.</summary>
    private static void AddPrompts(IDictionary<string, object> md, string dir)
    {
        string path = Path.Combine(dir, "config_sentence_transformers.json");
        if (!File.Exists(path))
        {
            return;
        }

        using var doc = JsonDocument.Parse(File.ReadAllBytes(path));
        var root = doc.RootElement;
        if (root.TryGetProperty("prompts", out var prompts) && prompts.ValueKind == JsonValueKind.Object)
        {
            var list = prompts.EnumerateObject()
                .Where(p => p.Value.ValueKind == JsonValueKind.String)
                .Select(p => $"{p.Name}={p.Value.GetString()}")
                .ToArray();
            if (list.Length > 0)
            {
                md[CanonicalKeys.EmbeddingPrompts] = list;
            }
        }

        if (root.TryGetProperty("default_prompt_name", out var defaultPrompt) && defaultPrompt.ValueKind == JsonValueKind.String)
        {
            md[CanonicalKeys.EmbeddingDefaultPrompt] = defaultPrompt.GetString()!;
        }
    }

    /// <summary>The tokenizer's maximum length from <c>sentence_bert_config.json</c>.</summary>
    private static void AddMaxSequenceLength(IDictionary<string, object> md, string dir)
    {
        string path = Path.Combine(dir, "sentence_bert_config.json");
        if (!File.Exists(path))
        {
            return;
        }

        using var doc = JsonDocument.Parse(File.ReadAllBytes(path));
        if (doc.RootElement.TryGetProperty("max_seq_length", out var maxLength) && maxLength.ValueKind == JsonValueKind.Number)
        {
            md[CanonicalKeys.TokenizerMaxLength] = maxLength.GetInt32();
        }
    }

    /// <summary>Cross-encoders: sentence-transformers applies a sigmoid to single-label outputs unless the model says otherwise.</summary>
    private static void AddCrossEncoder(IDictionary<string, object> md, HfConfig config)
    {
        if (!config.Architectures.Any(a => a.EndsWith("ForSequenceClassification", StringComparison.Ordinal)))
        {
            return;
        }

        string? activationFunction = config.GetString("sbert_ce_default_activation_function");
        var id2label = config.Get("id2label") is { ValueKind: JsonValueKind.Object } labelMap ? labelMap : (JsonElement?)null;
        int labelCount = id2label is { } map ? map.EnumerateObject().Count() : config.GetInt("num_labels") ?? 1;
        md[CanonicalKeys.ClassifierActivation] = activationFunction switch
        {
            null => labelCount == 1 ? "sigmoid" : "softmax",
            _ when activationFunction.EndsWith("Sigmoid", StringComparison.Ordinal) => "sigmoid",
            _ when activationFunction.EndsWith("Softmax", StringComparison.Ordinal) => "softmax",
            _ => "none",
        };
        if (id2label is { } labels)
        {
            md[CanonicalKeys.ClassifierLabels] = labels.EnumerateObject()
                .OrderBy(p => int.Parse(p.Name))
                .Select(p => p.Value.GetString() ?? "")
                .ToArray();
        }
    }

    private static string? Pooling(string path)
    {
        if (!File.Exists(path))
        {
            return null;
        }

        using var doc = JsonDocument.Parse(File.ReadAllBytes(path));
        var root = doc.RootElement;
        bool IsEnabled(string key) => root.TryGetProperty(key, out var value) && value.ValueKind == JsonValueKind.True;
        if (IsEnabled("pooling_mode_cls_token"))
        {
            return "cls";
        }

        if (IsEnabled("pooling_mode_lasttoken"))
        {
            return "last";
        }

        if (IsEnabled("pooling_mode_mean_tokens"))
        {
            return "mean";
        }

        if (root.TryGetProperty("pooling_mode", out var mode) && mode.ValueKind == JsonValueKind.String)
        {
            return mode.GetString();
        }

        throw new NotSupportedException($"{path}: only cls, mean and last-token pooling are supported.");
    }
}
