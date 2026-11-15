using System.Text.Json;
using System.Text.Json.Nodes;
using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Llm.Serving;
using Sharplight.Runtime;
using Sharplight.Vision;

namespace Sharplight.Families.Tests;

public partial class SpeculativeDecodingTests
{
    /// <summary>
    /// A Qwen3.5 model whose MTP layers live in a file of their own (unsloth ships its Qwen GGUFs with the NextN layers stripped and an
    /// <c>MTP/mtp-*.gguf</c> next to them): the tiny model split into a trunk without <c>mtp.*</c> and a sidecar with only <c>mtp.*</c>,
    /// the sidecar loaded as the <see cref="AuxiliaryRoles.Draft"/> source. The head proposes the drafts of the unsplit model, greedy
    /// output with it is token-identical, and the sidecar on its own is refused with a message that says how to use it.
    /// </summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task MtpSidecar_ProposesTheDraftsOfTheUnsplitModel(bool vulkan)
    {
        Assert.SkipUnless(Directory.Exists(ModelPath), $"{ModelPath} not present");
        string root = Path.Combine(Path.GetTempPath(), "sharplight-mtp-sidecar-" + Guid.NewGuid().ToString("N"));
        string trunk = Path.Combine(root, "trunk");
        string sidecar = Path.Combine(root, "sidecar");
        try
        {
            Split(ModelPath, trunk, sidecar, name => !name.StartsWith("mtp.", StringComparison.Ordinal));
            var error = Assert.Throws<NotSupportedException>(() => LlmModel.Load(sidecar, new LlmLoadOptions { Devices = [] }));
            Assert.Contains($"'{AuxiliaryRoles.Draft}' auxiliary source", error.Message, StringComparison.Ordinal);

            var whole = await DraftProposals(ModelPath, vulkan, []);
            var split = await DraftProposals(trunk, vulkan, [(AuxiliaryRoles.Draft, sidecar)]);
            Assert.NotEmpty(whole);
            Assert.Equal(whole, split);

            await Matches_NonSpeculativeDecoding(trunk, vulkan, "mtp", vulkan, SamplingParams.GreedyDefault,
                m => Assert.Equal(1, m.Definition.Properties[WellKnown.DraftLayers]), [(AuxiliaryRoles.Draft, sidecar)]);
        }
        finally
        {
            if (Directory.Exists(root))
            {
                Directory.Delete(root, recursive: true);
            }
        }
    }

    private static async Task<List<string>> DraftProposals(string modelPath, bool vulkan, IReadOnlyList<(string Role, string Path)> auxiliary)
    {
        using var device = OpenDevice(vulkan);
        using var model = LlmModel.Load(modelPath, new LlmLoadOptions
        {
            Devices = [new DeviceSlot { Device = device }], ContextLength = 512, MaxBatchTokens = 64, DeviceSampling = false, Auxiliary = auxiliary,
        });
        var decoder = model.CreateBatchDecoder(Slots, maxLogitsRows: Slots * (Drafts + 1), draftTokens: Drafts);
        var recorder = new RecordingSpeculator(model.CreateMtpSpeculator(Slots, Drafts));
        int[][] prompts = [[11, 22, 33, 44, 55, 66], [101, 202, 303, 404], [7, 8, 9, 7, 8, 9, 7, 8]];
        await Generate(model, decoder, recorder, prompts, SamplingParams.GreedyDefault, TestContext.Current.CancellationToken);
        return [.. recorder.Proposals.Select(p => string.Join(',', p.Context) + " -> " + string.Join(',', p.Drafts)).Order(StringComparer.Ordinal)];
    }

    /// <summary>Copies a single-file safetensors checkpoint into two directories (same small files), the tensors split by name.</summary>
    private static void Split(string source, string first, string second, Func<string, bool> inFirst)
    {
        Directory.CreateDirectory(first);
        Directory.CreateDirectory(second);
        foreach (string file in Directory.EnumerateFiles(source).Where(f => !f.EndsWith(".safetensors", StringComparison.Ordinal)))
        {
            File.Copy(file, Path.Combine(first, Path.GetFileName(file)));
            File.Copy(file, Path.Combine(second, Path.GetFileName(file)));
        }

        byte[] bytes = File.ReadAllBytes(Path.Combine(source, "model.safetensors"));
        int headerLength = checked((int)BitConverter.ToUInt64(bytes, 0));
        var header = JsonNode.Parse(bytes.AsSpan(8, headerLength))!.AsObject();
        int dataStart = 8 + headerLength;
        foreach (var (directory, keep) in new[] { (first, true), (second, false) })
        {
            var outHeader = new JsonObject();
            var data = new MemoryStream();
            foreach (var (name, node) in header)
            {
                if (name == "__metadata__")
                {
                    outHeader[name] = node!.DeepClone();
                    continue;
                }

                if (inFirst(name) != keep)
                {
                    continue;
                }

                var offsets = node!["data_offsets"]!.AsArray();
                long begin = offsets[0]!.GetValue<long>();
                long end = offsets[1]!.GetValue<long>();
                var entry = node.DeepClone().AsObject();
                entry["data_offsets"] = new JsonArray(data.Length, data.Length + end - begin);
                data.Write(bytes, dataStart + (int)begin, (int)(end - begin));
                outHeader[name] = entry;
            }

            byte[] json = JsonSerializer.SerializeToUtf8Bytes(outHeader);
            using var stream = File.Create(Path.Combine(directory, "model.safetensors"));
            stream.Write(BitConverter.GetBytes((ulong)json.Length));
            stream.Write(json);
            data.Position = 0;
            data.CopyTo(stream);
        }
    }
}
