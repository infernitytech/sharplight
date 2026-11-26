using System.Globalization;
using System.Text.Json;

namespace Sharplight.Hub;

/// <summary>Parses the Hub's JSON responses (search, revision info, tree listing) without reflection-based serialization.</summary>
internal static class HubJson
{
    /// <exception cref="HubException">The answer has no commit sha.</exception>
    public static HubRevisionInfo ParseRevision(string json)
    {
        using var document = Parse(json);
        var root = document.RootElement;
        var commit = String(root, "sha") ?? throw new HubException("The Hub's answer has no commit sha.");
        return new HubRevisionInfo(String(root, "id"), commit, IsGated(root), Architecture(root), CardLicense(root), BaseModel(root));
    }

    /// <summary>The model family the Hub derived from the files: <c>gguf.architecture</c>, else <c>config.model_type</c>.</summary>
    private static string? Architecture(JsonElement root)
    {
        foreach (var (section, key) in new[] { ("gguf", "architecture"), ("config", "model_type") })
        {
            if (root.TryGetProperty(section, out var value) && value.ValueKind == JsonValueKind.Object && String(value, key) is { Length: > 0 } name)
            {
                return name;
            }
        }

        return null;
    }

    /// <summary>The license the model card declares (<c>cardData.license</c>), else that of the first <c>license:</c> tag; null when neither is there.</summary>
    private static string? CardLicense(JsonElement root)
        => root.TryGetProperty("cardData", out var card) && card.ValueKind == JsonValueKind.Object && String(card, "license") is { Length: > 0 } license
            ? license
            : License(root);

    /// <summary>The model this repository was quantized from (the first <c>base_model:quantized:</c> tag), which marks it as a repackaging of that model; null otherwise.</summary>
    private static string? BaseModel(JsonElement root)
    {
        const string prefix = "base_model:quantized:";
        if (!root.TryGetProperty("tags", out var tags) || tags.ValueKind != JsonValueKind.Array)
        {
            return null;
        }

        foreach (var tag in tags.EnumerateArray())
        {
            if (tag.ValueKind == JsonValueKind.String && tag.GetString() is { } text && text.StartsWith(prefix, StringComparison.Ordinal) && text.Length > prefix.Length)
            {
                return text[prefix.Length..];
            }
        }

        return null;
    }

    public static IReadOnlyList<HubFile> ParseTree(string json)
    {
        using var document = Parse(json);
        var files = new List<HubFile>();
        foreach (var entry in document.RootElement.EnumerateArray())
        {
            if (String(entry, "type") != "file" || String(entry, "path") is not { } path)
            {
                continue;
            }

            long size = entry.TryGetProperty("size", out var sizeValue) && sizeValue.TryGetInt64(out long parsed) ? parsed : 0;
            // An LFS file's git oid is that of its pointer, not of the content; for a gated repository the caller may not download, the
            // Hub masks the LFS oid ("****..."): such a file has no checksum here (HubClient fills it from the revision's blob listing).
            bool isLfs = entry.TryGetProperty("lfs", out var lfs) && lfs.ValueKind == JsonValueKind.Object;
            string? sha256 = isLfs ? Sha256(String(lfs, "oid")) : null;
            string? gitOid = isLfs ? null : String(entry, "oid")?.ToLowerInvariant();
            files.Add(new HubFile(path, size, sha256, gitOid));
        }

        return files;
    }

    /// <summary>
    /// The SHA-256 of every LFS file in a revision info requested with <c>blobs=true</c> (<c>siblings[].lfs.sha256</c>), by path. The
    /// Hub lists them even where the tree listing masks them (gated repositories whose terms the caller has not accepted).
    /// </summary>
    public static IReadOnlyDictionary<string, string> ParseLfsHashes(string json)
    {
        using var document = Parse(json);
        var hashes = new Dictionary<string, string>(StringComparer.Ordinal);
        if (!document.RootElement.TryGetProperty("siblings", out var siblings) || siblings.ValueKind != JsonValueKind.Array)
        {
            return hashes;
        }

        foreach (var sibling in siblings.EnumerateArray())
        {
            if (String(sibling, "rfilename") is { } path && sibling.TryGetProperty("lfs", out var lfs) && lfs.ValueKind == JsonValueKind.Object
                && Sha256(String(lfs, "sha256")) is { } sha256)
            {
                hashes[path] = sha256;
            }
        }

        return hashes;
    }

    /// <summary>The lower-case form of a 64-digit hex SHA-256, or null for anything else (a masked or missing value).</summary>
    private static string? Sha256(string? value)
        => value is { Length: 64 } && value.All(char.IsAsciiHexDigit) ? value.ToLowerInvariant() : null;

    public static IReadOnlyList<HubModelSummary> ParseSearch(string json)
    {
        using var document = Parse(json);
        var results = new List<HubModelSummary>();
        foreach (var model in document.RootElement.EnumerateArray())
        {
            var id = String(model, "id") ?? String(model, "modelId");
            if (id is null)
            {
                continue;
            }

            var files = model.TryGetProperty("siblings", out var siblings) && siblings.ValueKind == JsonValueKind.Array
                ? siblings.EnumerateArray().Select(sibling => String(sibling, "rfilename")).OfType<string>().ToList()
                : [];
            var formats = Formats(files);
            var config = model.TryGetProperty("config", out var configValue) && configValue.ValueKind == JsonValueKind.Object ? configValue : default;
            var gguf = model.TryGetProperty("gguf", out var ggufValue) && ggufValue.ValueKind == JsonValueKind.Object ? ggufValue : default;
            results.Add(new HubModelSummary(
                id,
                Int64(model, "downloads") ?? 0,
                Int64(model, "likes") ?? 0,
                IsGated(model),
                String(model, "library_name"),
                String(model, "pipeline_tag"),
                Parameters(model),
                formats,
                ModelFileSelector.ListVariants(files),
                gguf.ValueKind == JsonValueKind.Object ? String(gguf, "architecture") : null,
                config.ValueKind == JsonValueKind.Object ? String(config, "model_type") : null,
                config.ValueKind == JsonValueKind.Object ? QuantizationMethod(config) : null,
                files.Any(GgufFileName.IsGguf) && files.Where(GgufFileName.IsGguf).All(GgufFileName.IsProjector),
                Date(model, "createdAt"),
                License(model),
                Date(model, "lastModified"),
                formats.Count == 0 ? RepoLayouts.UnsupportedFormat(files) : null));
        }

        return results;
    }

    /// <summary>The <c>quant_method</c> of a configuration's <c>quantization_config</c> (<c>mlx</c> for MLX's, which has none).</summary>
    private static string? QuantizationMethod(JsonElement config)
    {
        if (!config.TryGetProperty("quantization_config", out var quantization) || quantization.ValueKind != JsonValueKind.Object)
        {
            return null;
        }

        return String(quantization, "quant_method") ?? (quantization.TryGetProperty("bits", out _) ? "mlx" : null);
    }

    private static JsonDocument Parse(string json)
    {
        try
        {
            return JsonDocument.Parse(json);
        }
        catch (JsonException error)
        {
            throw new HubException($"The Hub sent an answer that is not valid JSON ({error.Message}).", inner: error);
        }
    }

    /// <summary>The id of the first <c>license:</c> tag, or null.</summary>
    private static string? License(JsonElement model)
    {
        if (!model.TryGetProperty("tags", out var tags) || tags.ValueKind != JsonValueKind.Array)
        {
            return null;
        }

        foreach (var tag in tags.EnumerateArray())
        {
            if (tag.ValueKind == JsonValueKind.String && tag.GetString() is { } text && text.StartsWith("license:", StringComparison.Ordinal))
            {
                return text["license:".Length..];
            }
        }

        return null;
    }

    private static IReadOnlyList<ModelFormat> Formats(IReadOnlyList<string> files)
    {
        var formats = new List<ModelFormat>();
        if (files.Any(file => GgufFileName.IsGguf(file) && !GgufFileName.IsProjector(file)))
        {
            formats.Add(ModelFormat.Gguf);
        }

        if (files.Any(file => file.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase)))
        {
            formats.Add(ModelFormat.Safetensors);
        }

        if (files.Any(file => file.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase)))
        {
            formats.Add(ModelFormat.Onnx);
        }

        if (files.Contains(RepoLayouts.DiffusersIndex))
        {
            formats.Add(ModelFormat.Diffusers);
        }

        if (files.Any(RepoLayouts.IsWhisperCpp))
        {
            formats.Add(ModelFormat.WhisperCpp);
        }

        if (files.Any(file => file.EndsWith(".slm", StringComparison.OrdinalIgnoreCase)))
        {
            formats.Add(ModelFormat.Slm);
        }

        // PyTorch weights beside safetensors, GGUF or a pipeline are copies or parts of that model, which is pulled in its other format.
        if (!formats.Any(format => format is ModelFormat.Safetensors or ModelFormat.Gguf or ModelFormat.Diffusers) && PyTorchSelection.HasWeights(files))
        {
            formats.Add(ModelFormat.PyTorch);
        }

        return formats;
    }

    /// <summary>The parameter count from the GGUF or safetensors metadata the Hub extracted.</summary>
    private static long? Parameters(JsonElement model)
    {
        foreach (var section in new[] { "gguf", "safetensors" })
        {
            if (model.TryGetProperty(section, out var metadata) && metadata.ValueKind == JsonValueKind.Object && Int64(metadata, "total") is { } total)
            {
                return total;
            }
        }

        return null;
    }

    /// <summary><c>gated</c> is <c>false</c> or the gating mode (<c>"auto"</c>, <c>"manual"</c>).</summary>
    private static bool IsGated(JsonElement element)
    {
        if (!element.TryGetProperty("gated", out var gated))
        {
            return false;
        }

        return gated.ValueKind == JsonValueKind.True || gated.ValueKind == JsonValueKind.String;
    }

    private static string? String(JsonElement element, string name)
    {
        return element.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.String ? value.GetString() : null;
    }

    private static DateTimeOffset? Date(JsonElement element, string name)
    {
        return element.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.String
            && DateTimeOffset.TryParse(value.GetString(), CultureInfo.InvariantCulture, DateTimeStyles.AssumeUniversal, out var date) ? date : null;
    }

    private static long? Int64(JsonElement element, string name)
    {
        return element.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.Number && value.TryGetInt64(out long result) ? result : null;
    }
}
