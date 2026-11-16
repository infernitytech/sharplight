using System.Text.Json;

namespace Sharplight.Hub;

/// <summary>What a Diffusers repository's <c>model_index.json</c> declares: the pipeline class and its components.</summary>
/// <param name="ClassName">The pipeline class (<c>StableDiffusionXLPipeline</c>, <c>FluxPipeline</c>, <c>WanPipeline</c>), or empty when not given.</param>
/// <param name="Components">The components the pipeline uses (the subfolders it loads), in file order; components set to null are left out.</param>
public sealed record DiffusersPipeline(string ClassName, IReadOnlyList<string> Components)
{
    /// <summary>Reads <c>model_index.json</c>.</summary>
    /// <param name="indexJson">The file's text.</param>
    /// <returns>The pipeline, or <see langword="null"/> when the text is not a pipeline index.</returns>
    public static DiffusersPipeline? Parse(string indexJson)
    {
        ArgumentNullException.ThrowIfNull(indexJson);
        try
        {
            using var document = JsonDocument.Parse(indexJson);
            if (document.RootElement.ValueKind != JsonValueKind.Object)
            {
                return null;
            }

            string className = "";
            var components = new List<string>();
            foreach (var property in document.RootElement.EnumerateObject())
            {
                if (property.Name == "_class_name")
                {
                    className = property.Value.ValueKind == JsonValueKind.String ? property.Value.GetString() ?? "" : "";
                }
                else if (!property.Name.StartsWith('_') && property.Value is { ValueKind: JsonValueKind.Array } entry
                    && entry.GetArrayLength() == 2 && entry[1].ValueKind == JsonValueKind.String)
                {
                    components.Add(property.Name);
                }
            }

            return new DiffusersPipeline(className, components);
        }
        catch (JsonException)
        {
            return null;
        }
    }
}
