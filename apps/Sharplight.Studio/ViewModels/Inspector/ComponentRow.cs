using Sharplight.Models;
using Sharplight.Package;
using Sharplight.Studio.Models;

namespace Sharplight.Studio.ViewModels.Inspector;

/// <summary>One component of the model description (<see cref="ModelComponent"/>).</summary>
/// <param name="Role">Component role (encoder, decoder, tokenizer, ...).</param>
/// <param name="Graphs">The component's graphs, comma separated; empty for configuration-only components.</param>
/// <param name="Weights">Count and size of the weights the component owns, or empty when it names no weight prefixes.</param>
/// <param name="Config">The configuration keys, comma separated.</param>
/// <param name="Attachments">The attachments with their sizes, comma separated.</param>
internal sealed record ComponentRow(string Role, string Graphs, string Weights, string Config, string Attachments)
{
    /// <summary>The row of a component.</summary>
    /// <param name="component">The component.</param>
    /// <param name="weights">The package's weights.</param>
    /// <param name="attachmentSizes">Bytes per stored attachment name.</param>
    public static ComponentRow From(ModelComponent component, IReadOnlyList<PackageWeightInfo> weights, IReadOnlyDictionary<string, long> attachmentSizes)
    {
        var owned = component.WeightPrefixes.Count == 0 ? [] : weights.Where(w => component.OwnsWeight(w.Name)).ToList();
        string weightText = owned.Count == 0 ? "" : $"{owned.Count:N0} · {ByteFormat.Format(owned.Sum(w => w.ByteSize))}";
        var attachments = component.Attachments.Select(a =>
            attachmentSizes.TryGetValue(a.Value, out long size) ? $"{a.Key} ({ByteFormat.Format(size)})" : $"{a.Key} (missing)");
        return new ComponentRow(component.Role, string.Join(", ", component.Graphs), weightText, string.Join(", ", component.Config.Keys),
            string.Join(", ", attachments));
    }
}
