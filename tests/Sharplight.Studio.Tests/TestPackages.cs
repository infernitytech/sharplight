using System.Buffers.Binary;
using System.Text;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Package;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Studio.Tests;

/// <summary>Writes small packages for the package inspector tests with <see cref="ModelPackage.Save"/>.</summary>
internal static class TestPackages
{
    /// <summary>Key of the first kernel of the code-tier set (content addressed).</summary>
    public const string CodeKernelKey = "kernel/00112233445566778899aabbccddeeff";

    /// <summary>
    /// A package with one graph (<c>decode</c>, 4 nodes over 2 layers), two weights, a state, properties, metadata (tokenizer and chat
    /// template) and two artifact sets (code tier with two kernels, binary tier with a pipeline cache).
    /// </summary>
    public static void WriteSmall(string path)
    {
        var weights = new Dictionary<string, WeightInfo>
        {
            ["w0"] = Weight("w0", DataType.F32, new Shape(4, 8)),
            ["w1"] = Weight("w1", DataType.F16, new Shape(4, 4)),
        };
        var state = new StateDefinition { Name = "kc", Type = DataType.F16, Shape = new Shape(Dim.Sym("ctx"), 4), Kind = StateKind.KeyCache, Group = "layers.0" };
        var b = new GraphBuilder("decode", n => weights.GetValueOrDefault(n));
        var x = b.Input("x", InputRole.Embeddings, DataType.F32, new Shape(Dim.Sym("T"), 8));
        Value y;
        using (b.Group("layers.0"))
        {
            y = b.MatMul(x, b.Weight("w0"), "proj");
            b.CacheWrite(b.State(state), y);
        }

        using (b.Group("layers.1"))
        {
            y = b.Silu(b.MatMul(y, b.Weight("w1"), "proj2"), "act");
        }

        b.Output("y", y);
        var definition = new ModelDefinition
        {
            Architecture = "test_arch",
            Name = "small",
            Weights = weights,
            Graphs = new Dictionary<string, ComputeGraph> { ["decode"] = b.Build() },
            States = [state],
            Properties = new Dictionary<string, object> { [WellKnown.VocabSize] = 3, [WellKnown.ContextLength] = 4096 },
            SourceMetadata = new Dictionary<string, object>
            {
                [CanonicalKeys.TokenizerTokens] = new[] { "a", "b", "<eos>" },
                [CanonicalKeys.TokenizerModel] = "gpt2",
                [CanonicalKeys.TokenizerEosId] = 2,
                [CanonicalKeys.ChatTemplate] = "{% for m in messages %}{{ m.content }}{% endfor %}",
                [CanonicalKeys.Name] = "small test model",
            },
        };

        var code = new CompilationCache(Descriptor(ArtifactTiers.Code, "spirv", (ArtifactDescriptorKeys.Target, "spirv_1_6"), (ArtifactDescriptorKeys.Subgroup, "32")));
        code.Put(CodeKernelKey, new byte[64]);
        code.Put("kernel/ffeeddccbbaa99887766554433221100", new byte[32]);
        var binary = new CompilationCache(Descriptor(ArtifactTiers.Binary, "pipeline-cache", (ArtifactDescriptorKeys.Vendor, "1234")));
        binary.Put("test/pipeline_cache", new byte[100]);
        ModelPackage.Save(path, definition, [code, binary]);
    }

    /// <summary>A package with one graph <c>main</c> of <paramref name="nodeCount"/> elementwise nodes over 100 layers.</summary>
    public static void WriteLarge(string path, int nodeCount)
    {
        var b = new GraphBuilder("main", _ => null);
        var y = b.Input("x", InputRole.Embeddings, DataType.F32, new Shape(Dim.Sym("T"), 64));
        int perLayer = Math.Max(1, nodeCount / 100);
        for (int i = 0; i < nodeCount; i++)
        {
            using (b.Group($"layers.{i / perLayer}"))
            {
                y = i % 2 == 0 ? b.Silu(y) : b.Add(y, y);
            }
        }

        b.Output("y", y);
        var definition = new ModelDefinition
        {
            Architecture = "test_large",
            Name = "large",
            Weights = new Dictionary<string, WeightInfo>(),
            Graphs = new Dictionary<string, ComputeGraph> { ["main"] = b.Build() },
            States = [],
            Properties = new Dictionary<string, object>(),
            SourceMetadata = new Dictionary<string, object>(),
        };
        ModelPackage.Save(path, definition, []);
    }

    /// <summary>
    /// Appends a section of a kind the format does not define (as a newer version might write) and a new section table that lists
    /// it after the existing sections; the old table stays behind as unused bytes.
    /// </summary>
    public static void AppendUnknownSection(string path, uint kind, string name, int length)
    {
        using var file = new FileStream(path, FileMode.Open, FileAccess.ReadWrite);
        Span<byte> header = stackalloc byte[32];
        file.ReadExactly(header);
        long table = BinaryPrimitives.ReadInt64LittleEndian(header[16..]);
        int count = BinaryPrimitives.ReadInt32LittleEndian(header[24..]);
        file.Seek(table, SeekOrigin.Begin);
        var entries = new List<(uint Kind, string Name, long Offset, long Length)>();
        using (var reader = new BinaryReader(file, Encoding.UTF8, leaveOpen: true))
        {
            for (int i = 0; i < count; i++)
            {
                uint k = reader.ReadUInt32();
                int nameLength = reader.ReadInt32();
                long offset = reader.ReadInt64();
                long size = reader.ReadInt64();
                entries.Add((k, Encoding.UTF8.GetString(reader.ReadBytes(nameLength)), offset, size));
            }
        }

        long sectionOffset = (file.Length + 4095) / 4096 * 4096;
        file.Seek(sectionOffset, SeekOrigin.Begin);
        file.Write(new byte[length]);
        entries.Add((kind, name, sectionOffset, length));
        long newTable = file.Position;
        using (var writer = new BinaryWriter(file, Encoding.UTF8, leaveOpen: true))
        {
            foreach (var entry in entries)
            {
                var bytes = Encoding.UTF8.GetBytes(entry.Name);
                writer.Write(entry.Kind);
                writer.Write(bytes.Length);
                writer.Write(entry.Offset);
                writer.Write(entry.Length);
                writer.Write(bytes);
            }

            writer.Seek(16, SeekOrigin.Begin);
            writer.Write(newTable);
            writer.Write(entries.Count);
        }
    }

    private static WeightInfo Weight(string name, DataType type, Shape shape) =>
        WeightInfo.Derived(name, type, shape, () => new byte[type.ByteSize(shape.Elements())]);

    private static Dictionary<string, string> Descriptor(string tier, string kind, params (string Key, string Value)[] more)
    {
        var descriptor = new Dictionary<string, string>
        {
            [ArtifactDescriptorKeys.Provider] = "testgpu",
            [ArtifactDescriptorKeys.Tier] = tier,
            [ArtifactDescriptorKeys.Kind] = kind,
        };
        foreach (var (key, value) in more)
        {
            descriptor[key] = value;
        }

        return descriptor;
    }
}
