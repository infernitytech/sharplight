using System.Text.Json;
using Sharplight.Imaging;
using Sharplight.Vision;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Translates a checkpoint's <c>preprocessor_config.json</c> (transformers' image processors: <c>size</c>, <c>resample</c>, <c>crop_size</c>,
/// <c>rescale_factor</c>, <c>image_mean</c>, <c>image_std</c>) into the neutral <see cref="ImageProcessorConfig"/> that is stored with the model.
/// </summary>
internal static class HfImageProcessorConfig
{
    /// <summary>The configuration of the checkpoint in <paramref name="directory"/>; transformers' ViT defaults (224², bilinear, mean and std 0.5) when it has no file.</summary>
    /// <param name="directory">The checkpoint directory.</param>
    /// <param name="defaultSize">The input size to use when the file gives none (the model configuration's image size).</param>
    public static ImageProcessorConfig Read(string directory, int defaultSize)
    {
        string path = Path.Combine(directory, "preprocessor_config.json");
        using var document = File.Exists(path) ? JsonDocument.Parse(File.ReadAllBytes(path)) : null;
        var root = document?.RootElement ?? default;
        bool Has(string key) => document is not null && root.TryGetProperty(key, out var v) && v.ValueKind != JsonValueKind.Null;
        bool Flag(string key, bool fallback) => Has(key) ? root.GetProperty(key).GetBoolean() : fallback;

        var config = new ImageProcessorConfig
        {
            ResizeMode = ImageResizeMode.Exact,
            Width = defaultSize,
            Height = defaultSize,
            Filter = Has("resample") ? Filter(root.GetProperty("resample").GetInt32()) : ImageResampleFilter.Bilinear,
            Rescale = Flag("do_rescale", true) ? (Has("rescale_factor") ? root.GetProperty("rescale_factor").GetSingle() : 1f / 255f) : 1f,
            Mean = Flag("do_normalize", true) ? Floats(root, Has("image_mean"), "image_mean", 0.5f) : [],
            Std = Flag("do_normalize", true) ? Floats(root, Has("image_std"), "image_std", 0.5f) : [],
        };
        if (!Flag("do_resize", true))
        {
            config = config with { ResizeMode = ImageResizeMode.None };
        }
        else if (Has("size"))
        {
            var size = root.GetProperty("size");
            config = size.ValueKind switch
            {
                // A bare number is the shortest edge for processors that center-crop afterwards (CLIP), a square otherwise.
                JsonValueKind.Number when Flag("do_center_crop", false) => config with { ResizeMode = ImageResizeMode.ShortestEdge, ShortestEdge = size.GetInt32() },
                JsonValueKind.Number => config with { Width = size.GetInt32(), Height = size.GetInt32() },
                _ when size.TryGetProperty("shortest_edge", out var shortest) => config with { ResizeMode = ImageResizeMode.ShortestEdge, ShortestEdge = shortest.GetInt32() },
                _ when size.TryGetProperty("longest_edge", out var longest) => config with { ResizeMode = ImageResizeMode.LongestEdge, LongestEdge = longest.GetInt32() },
                _ => config with { Width = size.GetProperty("width").GetInt32(), Height = size.GetProperty("height").GetInt32() },
            };
        }

        if (Flag("do_center_crop", false) && Has("crop_size"))
        {
            var crop = root.GetProperty("crop_size");
            config = crop.ValueKind == JsonValueKind.Number
                ? config with { CropWidth = crop.GetInt32(), CropHeight = crop.GetInt32() }
                : config with { CropWidth = crop.GetProperty("width").GetInt32(), CropHeight = crop.GetProperty("height").GetInt32() };
        }

        return config;
    }

    /// <summary>The side of the square model input the configuration produces for a square source: the center crop, the exact size, or <paramref name="fallback"/> for the aspect-preserving modes without crop.</summary>
    /// <param name="config">The configuration.</param>
    /// <param name="fallback">The size when the configuration does not fix one.</param>
    public static int InputSize(ImageProcessorConfig config, int fallback)
        => config.CropWidth > 0 ? config.CropWidth : config.ResizeMode == ImageResizeMode.Exact && config.Width > 0 ? config.Width : fallback;

    // PIL resampling constants: 0 nearest, 1 lanczos, 2 bilinear, 3 bicubic, 4 box, 5 hamming.
    private static ImageResampleFilter Filter(int resample) => resample switch
    {
        0 => ImageResampleFilter.Nearest,
        2 or 4 or 5 => ImageResampleFilter.Bilinear,
        _ => ImageResampleFilter.Bicubic,
    };

    private static float[] Floats(JsonElement root, bool present, string key, float fallback)
        => present ? [.. root.GetProperty(key).EnumerateArray().Select(v => v.GetSingle())] : [fallback, fallback, fallback];
}
