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
        return new HubRevisionInfo(String(root, "id"), commit, IsGated(root), Architecture(root));
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
            string? sha256 = entry.TryGetProperty("lfs", out var lfs) && lfs.ValueKind == JsonValueKind.Object ? String(lfs, "oid") : null;
            string? gitOid = sha256 is null ? String(entry, "oid") : null;
            files.Add(new HubFile(path, size, sha256?.ToLowerInvariant(), gitOid?.ToLowerInvariant()));
        }

        return files;
    }

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
            results.Add(new HubModelSummary(
                id,
                Int64(model, "downloads") ?? 0,
                Int64(model, "likes") ?? 0,
                IsGated(model),
                String(model, "library_name"),
                String(model, "pipeline_tag"),
                Parameters(model),
                Formats(files),
                ModelFileSelector.ListVariants(files)));
        }

        return results;
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

    private static long? Int64(JsonElement element, string name)
    {
        return element.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.Number && value.TryGetInt64(out long result) ? result : null;
    }
}
