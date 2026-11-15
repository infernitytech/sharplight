using Sharplight.Tensors;
using Xunit;

namespace Sharplight.Import.Gguf.Tests;

/// <summary>
/// <see cref="GgufSupport.Validate"/> runs the importer's mapping steps on a header alone. The fixture is the header of Unsloth's
/// gemma-4-26B-A4B-it UD-Q4_K_XL (made with <c>tools/make-gguf-header-fixture.py</c>: every key and tensor entry, the tokenizer arrays
/// emptied), a Gemma 4 mixture-of-experts file whose architecture was mapped while its expert tensors were not.
/// </summary>
public class GgufValidationTests
{
    private static readonly string s_gemma4Moe = Path.Combine(AppContext.BaseDirectory, "Fixtures", "gemma-4-26B-A4B-it-UD-Q4_K_XL.header.gguf");

    [Fact]
    public void Gemma4Moe_HeaderTensorTable_IsReadCompletely()
    {
        var header = File.ReadAllBytes(s_gemma4Moe);
        var table = GgufHeaderProbe.ReadTensorTable(header);

        Assert.NotNull(table);
        Assert.Equal(658, table.Count);
        Assert.Contains(table, tensor => tensor.Name == "blk.29.ffn_gate_up_exps.weight");
        Assert.Equal(new long[] { 2816, 1408, 128 }, table.Single(tensor => tensor.Name == "blk.0.ffn_gate_up_exps.weight").Dimensions);
        Assert.Equal(GgufHeaderProbe.CountTypes(table), GgufHeaderProbe.ReadTensorTypes(header));
        Assert.Null(GgufHeaderProbe.ReadTensorTable(header.AsSpan(0, header.Length - 1)));
    }

    [Fact]
    public void Gemma4Moe_Header_MapsLikeTheImporter()
    {
        var header = File.ReadAllBytes(s_gemma4Moe);
        var check = GgufSupport.Validate(GgufHeaderProbe.ReadMetadata(header), GgufHeaderProbe.ReadTensorTable(header)!);

        // The importer on the same header, padded (sparsely) to the size its tensor table needs: the probe must agree with it.
        string path = Path.Combine(Path.GetTempPath(), $"sharplight-gemma4-header-{Guid.NewGuid():N}.gguf");
        try
        {
            using (var file = File.Create(path))
            {
                file.Write(header);
                file.SetLength(32L << 30);
            }

            string? importError = null;
            try
            {
                var source = GgufImporter.Open(path);
                foreach (var resource in source.OwnedResources)
                {
                    resource.Dispose();
                }
            }
            catch (NotSupportedException error)
            {
                importError = error.Message;
            }

            Assert.True(check.IsComplete);
            Assert.Equal(importError is null, check.IsAccepted);
            if (importError is not null)
            {
                Assert.All(check.Problems, problem => Assert.Contains(problem, importError, StringComparison.Ordinal));
            }
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void UnmappedTensor_IsAProblem()
    {
        var header = File.ReadAllBytes(s_gemma4Moe);
        var table = GgufHeaderProbe.ReadTensorTable(header)!
            .Select(tensor => tensor.Name == "output_norm.weight" ? tensor with { Name = "output_norm_9.weight" } : tensor).ToList();

        var check = GgufSupport.Validate(GgufHeaderProbe.ReadMetadata(header), table);

        Assert.True(check.IsComplete);
        var problem = Assert.Single(check.Problems);
        Assert.Contains("output_norm_9.weight", problem, StringComparison.Ordinal);
        Assert.Contains("no canonical mapping", problem, StringComparison.Ordinal);
    }

    [Fact]
    public void RejectedHyperParameter_IsAProblem()
    {
        var metadata = new Dictionary<string, object>
        {
            ["general.architecture"] = "nemotron_h_moe",
            ["nemotron_h_moe.expert_group_count"] = 4u,
        };

        var check = GgufSupport.Validate(metadata, [new GgufTensorEntry("token_embd.weight", 0, [8, 16])]);

        Assert.True(check.IsComplete);
        Assert.Contains(check.Problems, problem => problem.Contains("group-limited expert routing", StringComparison.Ordinal));
    }

    [Fact]
    public void UnknownArchitectureOrTypeId_IsAProblem()
    {
        var unmapped = GgufSupport.Validate(new Dictionary<string, object> { ["general.architecture"] = "toy_unmapped" }, []);
        Assert.Contains("toy_unmapped", Assert.Single(unmapped.Problems), StringComparison.Ordinal);

        var header = File.ReadAllBytes(s_gemma4Moe);
        var table = GgufHeaderProbe.ReadTensorTable(header)!.Select((tensor, i) => i == 657 ? tensor with { GgmlType = 99 } : tensor).ToList();
        var check = GgufSupport.Validate(GgufHeaderProbe.ReadMetadata(header), table);
        Assert.Contains("99", Assert.Single(check.Problems), StringComparison.Ordinal);
    }

    [Fact]
    public void WrittenFile_MapsFromItsHeader()
    {
        using var stream = new MemoryStream();
        new GgufWriter()
            .AddMetadata("general.architecture", "llama")
            .AddMetadata("llama.block_count", 1u)
            .AddMetadata("llama.embedding_length", 4u)
            .AddMetadata("llama.feed_forward_length", 8u)
            .AddMetadata("llama.attention.head_count", 1u)
            .AddMetadata("llama.attention.layer_norm_rms_epsilon", 1e-5f)
            .AddTensor("token_embd.weight", DataType.F32, new byte[4 * 4 * 4], 4, 4)
            .AddTensor("output_norm.weight", DataType.F32, new byte[16], 4)
            .Write(stream);
        var bytes = stream.ToArray();

        var check = GgufSupport.Validate(GgufHeaderProbe.ReadMetadata(bytes), GgufHeaderProbe.ReadTensorTable(bytes)!);

        Assert.True(check.IsAccepted, string.Join(" ", check.Problems));
    }
}
