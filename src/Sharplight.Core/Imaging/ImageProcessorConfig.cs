using System.Text.Json;
using System.Text.Json.Nodes;
using Sharplight.Vision;

namespace Sharplight.Imaging;

/// <summary>
/// The host-side preprocessing of an image model, in a format-neutral form: resize, center crop, rescale and normalize. Importers
/// translate the model's own processor configuration into it; it is stored with the model (the <c>imageProcessor</c> key of the
/// feature-extractor component's configuration, see <see cref="ImageProcessor.ComponentKey"/>) and applied by <see cref="ImageProcessor"/>.
/// </summary>
public sealed record ImageProcessorConfig
{
    /// <summary>How the image is resized.</summary>
    public ImageResizeMode ResizeMode { get; init; } = ImageResizeMode.Exact;

    /// <summary>Target width for <see cref="ImageResizeMode.Exact"/>.</summary>
    public int Width { get; init; }

    /// <summary>Target height for <see cref="ImageResizeMode.Exact"/>.</summary>
    public int Height { get; init; }

    /// <summary>Target of the shorter edge for <see cref="ImageResizeMode.ShortestEdge"/>.</summary>
    public int ShortestEdge { get; init; }

    /// <summary>Target of the longer edge for <see cref="ImageResizeMode.LongestEdge"/>.</summary>
    public int LongestEdge { get; init; }

    /// <summary>The resampling filter.</summary>
    public ImageResampleFilter Filter { get; init; } = ImageResampleFilter.Bilinear;

    /// <summary>Width of the center crop after resizing, or 0 for none.</summary>
    public int CropWidth { get; init; }

    /// <summary>Height of the center crop after resizing, or 0 for none.</summary>
    public int CropHeight { get; init; }

    /// <summary>Multiplier that maps 0..255 to the model's raw range (1/255 for models that expect 0..1).</summary>
    public float Rescale { get; init; } = 1f / 255f;

    /// <summary>Per-channel mean subtracted after rescaling (3 values), or empty for none.</summary>
    public IReadOnlyList<float> Mean { get; init; } = [];

    /// <summary>Per-channel standard deviation divided by after the mean (3 values), or empty for none.</summary>
    public IReadOnlyList<float> Std { get; init; } = [];

    /// <summary>The configuration as a JSON object (camelCase, enums as lowercase hyphenated text).</summary>
    public JsonObject ToJson() => new()
    {
        ["resizeMode"] = ResizeMode switch
        {
            ImageResizeMode.None => "none",
            ImageResizeMode.Exact => "exact",
            ImageResizeMode.ShortestEdge => "shortest-edge",
            _ => "longest-edge",
        },
        ["width"] = Width,
        ["height"] = Height,
        ["shortestEdge"] = ShortestEdge,
        ["longestEdge"] = LongestEdge,
        ["filter"] = Filter.ToString().ToLowerInvariant(),
        ["cropWidth"] = CropWidth,
        ["cropHeight"] = CropHeight,
        ["rescale"] = Rescale,
        ["mean"] = new JsonArray([.. Mean.Select(v => (JsonNode?)JsonValue.Create(v))]),
        ["std"] = new JsonArray([.. Std.Select(v => (JsonNode?)JsonValue.Create(v))]),
    };

    /// <summary>Reads a configuration written by <see cref="ToJson"/>; absent keys keep their defaults.</summary>
    /// <param name="json">A JSON object.</param>
    /// <exception cref="InvalidDataException">The value is not an object or has an unknown mode or filter.</exception>
    public static ImageProcessorConfig FromJson(JsonElement json)
    {
        if (json.ValueKind != JsonValueKind.Object)
        {
            throw new InvalidDataException("An image processor configuration must be a JSON object.");
        }

        var result = new ImageProcessorConfig();
        foreach (var property in json.EnumerateObject())
        {
            var v = property.Value;
            result = property.Name switch
            {
                "resizeMode" => result with
                {
                    ResizeMode = v.GetString() switch
                    {
                        "none" => ImageResizeMode.None,
                        "exact" => ImageResizeMode.Exact,
                        "shortest-edge" => ImageResizeMode.ShortestEdge,
                        "longest-edge" => ImageResizeMode.LongestEdge,
                        var other => throw new InvalidDataException($"Unknown image resize mode '{other}'."),
                    },
                },
                "width" => result with { Width = v.GetInt32() },
                "height" => result with { Height = v.GetInt32() },
                "shortestEdge" => result with { ShortestEdge = v.GetInt32() },
                "longestEdge" => result with { LongestEdge = v.GetInt32() },
                "filter" => result with
                {
                    Filter = Enum.TryParse<ImageResampleFilter>(v.GetString(), true, out var filter)
                        ? filter
                        : throw new InvalidDataException($"Unknown image resample filter '{v.GetString()}'."),
                },
                "cropWidth" => result with { CropWidth = v.GetInt32() },
                "cropHeight" => result with { CropHeight = v.GetInt32() },
                "rescale" => result with { Rescale = v.GetSingle() },
                "mean" => result with { Mean = [.. v.EnumerateArray().Select(e => e.GetSingle())] },
                "std" => result with { Std = [.. v.EnumerateArray().Select(e => e.GetSingle())] },
                _ => result,
            };
        }

        return result;
    }
}
