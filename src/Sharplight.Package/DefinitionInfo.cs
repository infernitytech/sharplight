using System.Collections;
using System.Globalization;
using Sharplight.Models;

namespace Sharplight.Package;

/// <summary>Summary of a package's stored model definition (Sharplight IR).</summary>
public sealed class DefinitionInfo
{
    internal DefinitionInfo(ModelDefinition definition)
    {
        Architecture = definition.Architecture;
        Name = definition.Name;
        Graphs = [.. definition.Graphs.OrderBy(g => g.Key, StringComparer.Ordinal).Select(g => new GraphInfo(g.Key, g.Value))];
        States = definition.States;
        Properties = definition.Properties;
        Metadata = definition.SourceMetadata;
    }

    /// <summary>Canonical architecture id.</summary>
    public string Architecture { get; }

    /// <summary>Model name.</summary>
    public string Name { get; }

    /// <summary>Stored graphs, by name.</summary>
    public IReadOnlyList<GraphInfo> Graphs { get; }

    /// <summary>Persistent states (KV caches, recurrent state).</summary>
    public IReadOnlyList<StateDefinition> States { get; }

    /// <summary>Model properties the runtime reads (vocabulary size, context length, ...), raw values.</summary>
    public IReadOnlyDictionary<string, object> Properties { get; }

    /// <summary>Source metadata (tokenizer, chat template, ...), raw values; arrays can be large (vocabularies).</summary>
    public IReadOnlyDictionary<string, object> Metadata { get; }

    /// <summary>Formats a property or metadata value for display: arrays as their length and first items, long strings shortened.</summary>
    /// <param name="value">A raw value from <see cref="Properties"/> or <see cref="Metadata"/>.</param>
    /// <param name="maxItems">Array items shown before the rest is elided.</param>
    /// <param name="maxLength">Characters shown of a string before the rest is elided.</param>
    public static string FormatValue(object value, int maxItems = 8, int maxLength = 120)
    {
        if (value is string text)
        {
            var oneLine = text.ReplaceLineEndings("\\n");
            return oneLine.Length <= maxLength ? oneLine : $"{oneLine[..maxLength]}... ({text.Length} chars)";
        }

        if (value is IList list)
        {
            var items = list.Cast<object>().Take(maxItems).Select(v => FormatValue(v, maxItems, maxLength));
            var more = list.Count > maxItems ? ", ..." : "";
            return $"[{list.Count}: {string.Join(", ", items)}{more}]";
        }

        return Convert.ToString(value, CultureInfo.InvariantCulture) ?? "";
    }
}
