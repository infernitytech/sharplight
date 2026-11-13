using System.Text.Json;
using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>
/// Translates sentence-transformers metadata stored next to a checkpoint (<c>modules.json</c>, the Pooling module's config,
/// <c>config_sentence_transformers.json</c>, <c>sentence_bert_config.json</c>) and cross-encoder hints in <c>config.json</c> into the
/// canonical <c>embedding.*</c> / <c>classifier.*</c> keys.
/// </summary>
public static class HfSentenceTransformers
{
    /// <summary>Adds the canonical embedding / classifier keys found in the sentence-transformers files of the model directory to <paramref name="md"/> (<paramref name="dir"/>: the model directory).</summary>
    public static void AddTo(IDictionary<string, object> md, string dir, HfConfig config)
    {
        string modules = Path.Combine(dir, "modules.json");
        if (File.Exists(modules))
        {
            using var doc = JsonDocument.Parse(File.ReadAllBytes(modules));
            foreach (var m in doc.RootElement.EnumerateArray())
            {
                string type = m.TryGetProperty("type", out var t) ? t.GetString() ?? "" : "";
                string path = m.TryGetProperty("path", out var p) ? p.GetString() ?? "" : "";
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
                    throw new NotSupportedException($"{dir}: sentence-transformers Dense modules after pooling are not supported.");
                }
            }
            if (!md.ContainsKey(CanonicalKeys.EmbeddingNormalize))
            {
                md[CanonicalKeys.EmbeddingNormalize] = false;
            }
        }

        string stConfig = Path.Combine(dir, "config_sentence_transformers.json");
        if (File.Exists(stConfig))
        {
            using var doc = JsonDocument.Parse(File.ReadAllBytes(stConfig));
            var root = doc.RootElement;
            if (root.TryGetProperty("prompts", out var prompts) && prompts.ValueKind == JsonValueKind.Object)
            {
                var list = prompts.EnumerateObject().Where(p => p.Value.ValueKind == JsonValueKind.String).Select(p => $"{p.Name}={p.Value.GetString()}").ToArray();
                if (list.Length > 0)
                {
                    md[CanonicalKeys.EmbeddingPrompts] = list;
                }
            }
            if (root.TryGetProperty("default_prompt_name", out var dp) && dp.ValueKind == JsonValueKind.String)
            {
                md[CanonicalKeys.EmbeddingDefaultPrompt] = dp.GetString()!;
            }
        }

        string sbert = Path.Combine(dir, "sentence_bert_config.json");
        if (File.Exists(sbert))
        {
            using var doc = JsonDocument.Parse(File.ReadAllBytes(sbert));
            if (doc.RootElement.TryGetProperty("max_seq_length", out var ml) && ml.ValueKind == JsonValueKind.Number)
            {
                md[CanonicalKeys.TokenizerMaxLength] = ml.GetInt32();
            }
        }

        // Cross-encoders: sentence-transformers applies a sigmoid to single-label outputs unless the model says otherwise.
        if (config.Architectures.Any(a => a.EndsWith("ForSequenceClassification", StringComparison.Ordinal)))
        {
            string? fn = config.GetString("sbert_ce_default_activation_function");
            int labels = config.Get("id2label") is { ValueKind: JsonValueKind.Object } l ? l.EnumerateObject().Count() : config.GetInt("num_labels") ?? 1;
            md[CanonicalKeys.ClassifierActivation] = fn switch
            {
                null => labels == 1 ? "sigmoid" : "softmax",
                _ when fn.EndsWith("Sigmoid", StringComparison.Ordinal) => "sigmoid",
                _ when fn.EndsWith("Softmax", StringComparison.Ordinal) => "softmax",
                _ => "none",
            };
            if (config.Get("id2label") is { ValueKind: JsonValueKind.Object } id2)
            {
                md[CanonicalKeys.ClassifierLabels] = id2.EnumerateObject().OrderBy(p => int.Parse(p.Name)).Select(p => p.Value.GetString() ?? "").ToArray();
            }
        }
    }

    private static string? Pooling(string path)
    {
        if (!File.Exists(path))
        {
            return null;
        }

        using var doc = JsonDocument.Parse(File.ReadAllBytes(path));
        var r = doc.RootElement;
        bool On(string k) => r.TryGetProperty(k, out var v) && v.ValueKind == JsonValueKind.True;
        if (On("pooling_mode_cls_token"))
        {
            return "cls";
        }

        if (On("pooling_mode_lasttoken"))
        {
            return "last";
        }

        if (On("pooling_mode_mean_tokens"))
        {
            return "mean";
        }

        if (r.TryGetProperty("pooling_mode", out var pm) && pm.ValueKind == JsonValueKind.String)
        {
            return pm.GetString();
        }

        throw new NotSupportedException($"{path}: only cls, mean and last-token pooling are supported.");
    }
}
