using Sharplight.Graph;
using Sharplight.Runtime;
using Xunit;

namespace Sharplight.Package.Tests;

/// <summary>The parts of the inspection API that viewers use to show the whole file: layout, dataflow edges and entry offsets.</summary>
public class InspectionApiTests
{
    [Fact]
    public void SectionKindsListEveryKindOfTheFormat()
    {
        Assert.Equal(["DEFN", "WIDX", "WDAT", "KSET", "ARTS", "DESC", "ATCH"], ModelPackageReader.SectionKinds);
    }

    [Fact]
    public void HeaderSectionsAndTableCoverTheFile()
    {
        using var tmp = new TempDir();
        var p = tmp.File("a.slm");
        var code = new CompilationCache(FakeArtifactDevice.Code());
        code.Put("kernel/1", new byte[] { 1, 2, 3 });
        code.Put("kernel/2", new byte[40]);
        ModelPackage.Save(p, PackageTests.SmallDefinition(), [code]);
        var info = ModelPackageReader.Open(p);

        Assert.Equal(64, info.HeaderLength);
        Assert.True(info.SectionTableOffset >= info.Sections.Max(s => s.Offset + s.Length));
        Assert.Equal(info.FileSize, info.SectionTableOffset + info.SectionTableLength);
        Assert.All(info.Sections, s => Assert.Contains(s.Kind, ModelPackageReader.SectionKinds));

        var set = Assert.Single(info.ArtifactSets);
        foreach (var entry in set.Entries)
        {
            Assert.InRange(entry.Offset, set.Offset, set.Offset + set.Length - entry.Size);
        }

        Assert.NotEqual(set.Entries[0].Offset, set.Entries[1].Offset);
    }

    [Fact]
    public void NodeInputsReferenceTheirProducers()
    {
        using var tmp = new TempDir();
        var p = tmp.File("a.slm");
        ModelPackage.Save(p, PackageTests.SmallDefinition(), []);
        var graph = ModelPackageReader.Open(p).Definition.Graphs.Single();
        var ids = graph.Nodes.Select(n => n.Id).ToHashSet();
        foreach (var node in graph.Nodes)
        {
            Assert.Equal(node.Inputs.Count, node.InputValues.Count);
            foreach (var input in node.InputValues)
            {
                if (input.Kind == ValueKind.Intermediate)
                {
                    Assert.Contains(input.Producer, ids);
                    Assert.True(input.Producer < node.Id || graph.Nodes.Select(n => n.Id).ToList().IndexOf(input.Producer) < graph.Nodes.ToList().IndexOf(node));
                }
                else
                {
                    Assert.Equal(-1, input.Producer);
                }
            }
        }

        Assert.Contains(graph.Nodes.SelectMany(n => n.InputValues), v => v.Kind == ValueKind.Weight);
        Assert.Contains(graph.Nodes.SelectMany(n => n.InputValues), v => v.Kind == ValueKind.Input);
    }
}
