using Xunit;

namespace Sharplight.Gguf.Tests;

/// <summary>
/// Headers of released checkpoints that the round-10 sweep found rejected by the importer, each fixed since (made with
/// <c>tools/make-gguf-header-fixture.py</c>; the split models with every shard's tensor entries): <see cref="GgufSupport.Validate"/> now
/// accepts each of them completely, and the importer agrees on the single-file one.
/// </summary>
public class GgufSweepFixtureTests
{
    private static string Fixture(string name) => Path.Combine(AppContext.BaseDirectory, "Fixtures", name + ".header.gguf");

    [Theory]
    [InlineData("DeepSeek-V4.1-Flash-1017-UD-Q2_K_XL")] // the vision-language expert bias exp_probs_b_vl
    [InlineData("DeepSeek-V4-Flash-0731-UD-Q2_K_XL")]
    [InlineData("Qwen3.8-Flash-Next-UD-IQ1_M")] // the n-gram convolution stored in BF16
    [InlineData("Llama-3.2-1B-Instruct-Q4_0_4_4")] // the interleaved Q4_0 type 31
    [InlineData("embeddinggemma-2-Q8_0")] // the gemma-embedding2 architecture, mapped since
    [InlineData("NVIDIA-Nemotron-3.5-Bolt-74B-A9B-UD-IQ1_M")] // the latent experts (ffn_latent_down / ffn_latent_up) of nemotron_h_moe
    public void ReleasedHeader_IsAccepted(string name)
    {
        var header = File.ReadAllBytes(Fixture(name));
        var table = GgufHeaderProbe.ReadTensorTable(header);
        Assert.NotNull(table);

        var check = GgufSupport.Validate(GgufHeaderProbe.ReadMetadata(header), table);

        Assert.True(check.IsComplete);
        Assert.True(check.IsAccepted, string.Join(" ", check.Problems));
    }

    /// <summary>
    /// Unsloth's split MTP file of Qwen3.8-27B (<c>MTP/mtp-Qwen3.8-27B-Q4_0.gguf</c>, architecture qwen35) holds only the
    /// multi-token-prediction layers: the check reports a draft model, as the importer does, instead of accepting it.
    /// </summary>
    [Fact]
    public void MtpOnlyHeader_IsADraftModel()
    {
        var header = File.ReadAllBytes(Fixture("mtp-Qwen3.8-27B-Q4_0"));

        var check = GgufSupport.Validate(GgufHeaderProbe.ReadMetadata(header), GgufHeaderProbe.ReadTensorTable(header)!);

        Assert.True(check.IsDraftModel);
        Assert.False(check.IsAccepted);
        Assert.Contains("multi-token-prediction", Assert.Single(check.Problems), StringComparison.Ordinal);
        Assert.False(GgufSupport.Validate(GgufHeaderProbe.ReadMetadata(File.ReadAllBytes(Fixture("Qwen3.8-Flash-Next-UD-IQ1_M"))),
            GgufHeaderProbe.ReadTensorTable(File.ReadAllBytes(Fixture("Qwen3.8-Flash-Next-UD-IQ1_M")))!).IsDraftModel);
    }

    [Fact]
    public void InterleavedQ4_0Header_ImportsFromItsSparseFile()
    {
        var header = File.ReadAllBytes(Fixture("Llama-3.2-1B-Instruct-Q4_0_4_4"));
        string path = Path.Combine(Path.GetTempPath(), $"sharplight-q4_0_4_4-header-{Guid.NewGuid():N}.gguf");
        try
        {
            using (var file = File.Create(path))
            {
                file.Write(header);
                file.SetLength(2L << 30);
            }

            var source = GgufImporter.Open(path);
            try
            {
                Assert.Contains(source.Weights.Values, weight => weight.Type.Name == "q4_0");
            }
            finally
            {
                foreach (var resource in source.OwnedResources)
                {
                    resource.Dispose();
                }
            }
        }
        finally
        {
            File.Delete(path);
        }
    }
}
