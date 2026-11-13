using Sharplight.Package;
using Sharplight.Runtime;
using Sharplight.Samples;

namespace Sharplight.Convert;

/// <summary>Writes the human-readable report of <c>info</c>.</summary>
internal sealed class PackageReport(PackageInfo info, TextWriter output)
{
    private const int HistogramOps = 12;
    private const int MetadataItems = 6;

    public void Write(bool listWeights, bool listEntries)
    {
        output.WriteLine($"{info.Path}: Sharplight package version {info.Version}, {ByteSize.Format(info.FileSize)}"
            + (info.ReclaimableBytes > 0 ? $" ({ByteSize.Format(info.ReclaimableBytes)} reclaimable by compact)" : ""));
        WriteSections();
        WriteDefinition();
        WriteWeights(listWeights);
        WriteArtifactSets(listEntries);
    }

    /// <exception cref="SampleException">The package has no graph named <paramref name="graphName"/>.</exception>
    public void WriteNodes(string graphName)
    {
        var graph = info.Definition.Graphs.FirstOrDefault(g => g.Name == graphName)
            ?? throw new SampleException($"The package has no graph '{graphName}'; graphs: {string.Join(", ", info.Definition.Graphs.Select(g => g.Name))}.");
        output.WriteLine();
        output.WriteLine($"Nodes of {graph.Name}:");
        foreach (var node in graph.Nodes)
        {
            var outputs = string.Join(", ", node.Outputs.Select(o => $"{o.Type}{o.Shape}"));
            output.WriteLine($"  #{node.Id,-5} {node.Op,-22} {outputs,-40} {node.Group,-10} {node.Attributes}");
        }
    }

    private void WriteSections()
    {
        output.WriteLine();
        output.WriteLine("Sections:");
        foreach (var s in info.Sections)
        {
            output.WriteLine($"  {s.Kind}  offset {s.Offset,14:N0}  {ByteSize.Format(s.Length),10}  {s.Name}");
        }
    }

    private void WriteDefinition()
    {
        var definition = info.Definition;
        output.WriteLine();
        output.WriteLine($"Model: {definition.Name} ({definition.Architecture}), {definition.States.Count} state(s)");
        foreach (var graph in definition.Graphs)
        {
            output.WriteLine($"  graph {graph.Name}: {graph.NodeCount} nodes, {graph.WeightCount} weights ({ByteSize.Format(graph.WeightBytes)}), {graph.StateCount} states");
            foreach (var input in graph.Inputs)
            {
                output.WriteLine($"    input  {input}");
            }

            foreach (var result in graph.Outputs)
            {
                output.WriteLine($"    output {result}");
            }

            var ops = graph.OpHistogram.Take(HistogramOps).Select(kv => $"{kv.Key} {kv.Value}");
            var more = graph.OpHistogram.Count > HistogramOps ? $", ... ({graph.OpHistogram.Count} ops)" : "";
            output.WriteLine($"    ops    {string.Join(", ", ops)}{more}");
        }

        WriteValues("Properties", definition.Properties);
        WriteValues("Metadata", definition.Metadata);
    }

    private void WriteValues(string title, IReadOnlyDictionary<string, object> values)
    {
        output.WriteLine($"  {title}:");
        foreach (var key in values.Keys.Order(StringComparer.Ordinal))
        {
            output.WriteLine($"    {key} = {DefinitionInfo.FormatValue(values[key], MetadataItems)}");
        }
    }

    private void WriteWeights(bool list)
    {
        output.WriteLine();
        output.WriteLine($"Weights: {info.Weights.Count}, {ByteSize.Format(info.WeightBytes)}");
        var byType = info.Weights.GroupBy(w => w.Type.Name).OrderByDescending(g => g.Sum(w => w.ByteSize));
        output.WriteLine("  " + string.Join(", ", byType.Select(g => $"{g.Key}: {g.Count()} ({ByteSize.Format(g.Sum(w => w.ByteSize))})")));
        if (!list)
        {
            return;
        }

        foreach (var w in info.Weights)
        {
            output.WriteLine($"  {w.Name,-48} {w.Type.Name,-6} {w.Shape,-24} {ByteSize.Format(w.ByteSize),10}");
        }
    }

    private void WriteArtifactSets(bool listEntries)
    {
        output.WriteLine();
        if (info.ArtifactSets.Count == 0)
        {
            output.WriteLine("Artifact sets: none (kernels are compiled at load)");
            return;
        }

        output.WriteLine($"Artifact sets: {info.ArtifactSets.Count}");
        foreach (var group in info.ArtifactSets.GroupBy(s => (s.Provider, s.Tier)))
        {
            foreach (var set in group)
            {
                var legacy = set.IsLegacy ? " (version 1 kernel set)" : "";
                output.WriteLine($"  {set.Provider}/{set.Tier}: {set.EntryCount} entries, {ByteSize.Format(set.TotalBytes)}{legacy}");
                output.WriteLine($"    {ArtifactDescriptor.Format(set.Descriptor)}");
                if (listEntries)
                {
                    foreach (var entry in set.Entries)
                    {
                        output.WriteLine($"    {entry.Key,-60} {ByteSize.Format(entry.Size),10}");
                    }
                }
            }
        }
    }
}
