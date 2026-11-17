using System.Globalization;
using Sharplight.Package;

namespace Sharplight.FeedTool;

/// <summary>Opens a built package with the engine's reader (no weights are loaded) and reads what the manifest needs.</summary>
internal static class PackageInspector
{
    private static readonly string[] ContextKeys = ["context_length", "max_position_embeddings", "n_ctx_train", "max_context", "n_text_ctx"];

    /// <summary>Inspects <paramref name="path"/>.</summary>
    /// <exception cref="InvalidDataException">The file is not a readable package.</exception>
    public static PackageInspection Inspect(string path)
    {
        var info = ModelPackageReader.Open(path);
        var definition = info.Definition;
        var properties = definition.Properties.Concat(definition.Metadata).ToList();
        var parameters = info.Weights.Sum(weight => weight.Shape.Elements());
        var template = properties.Any(property => property.Key.Contains("chat_template", StringComparison.OrdinalIgnoreCase));
        return new PackageInspection(
            string.IsNullOrWhiteSpace(definition.Name) ? Path.GetFileNameWithoutExtension(path) : definition.Name,
            definition.HasDescription ? definition.Description.Kind : "text-generation",
            string.IsNullOrWhiteSpace(definition.Architecture) ? null : definition.Architecture,
            ContextLength(properties),
            parameters,
            info.WeightBytes,
            info.Version,
            template);
    }

    private static int? ContextLength(IEnumerable<KeyValuePair<string, object>> properties)
    {
        foreach (var (key, value) in properties)
        {
            if (ContextKeys.Any(candidate => key.EndsWith(candidate, StringComparison.OrdinalIgnoreCase))
                && long.TryParse(Convert.ToString(value, CultureInfo.InvariantCulture), NumberStyles.Integer, CultureInfo.InvariantCulture, out var length)
                && length is > 0 and <= int.MaxValue)
            {
                return (int)length;
            }
        }

        return null;
    }
}
