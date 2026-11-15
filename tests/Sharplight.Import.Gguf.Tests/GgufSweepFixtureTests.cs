using Xunit;

namespace Sharplight.Import.Gguf.Tests;

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
    [InlineData("NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0")] // the NextN block appended after the trunk (blk.52.nextn.*)
    [InlineData("Inkling-Small-UD-IQ1_S")] // the inkling architecture (unsloth's fork), 3 shards
    [InlineData("Kimi-K3-UD-TQ1_0")] // the fork's narrow IQ1 type 65, read as IQ1_S, 12 shards
    [InlineData("Qwen3.8-Flash-Next-Gyro-S-TQ1_0")] // Gyro's trellis types 145 / 146 on Hadamard-rotated experts
    [InlineData("Qwen3.8-Flash-Next-Gyro-M-TQ2_0")] // trellis type 144
    [InlineData("Qwen3.8-Flash-Next-Gyro-L-TQ3_0")] // trellis types 147-149
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
    /// <remarks>
    /// Also ggml-org's Nemotron 3.5 Lightning MTP file (nemotron_h_moe, its NextN block below <c>mtp.layers.0.</c>) and agentionai's
    /// Qwen3.8-Flash-Next MTP file (qwen4exp, whose NextN head is not imported, so the file maps to no layers at all: the check
    /// recognizes it from the block indices, and the importer says so instead of failing on a missing trunk weight).
    /// </remarks>
    /// <summary>
    /// agentionai's <c>-noPLE</c> Gyro files leave out the n-gram table, which the fork loads from a separate file (<c>--ple</c>): the check
    /// reports that, as the importer does, instead of accepting a file whose import stops at the missing table.
    /// </summary>
    [Fact]
    public void NoPleHeader_IsRejectedForItsSeparateTable()
    {
        var header = File.ReadAllBytes(Fixture("Qwen3.8-Flash-Next-Gyro-S-TQ1_0-noPLE"));

        var check = GgufSupport.Validate(GgufHeaderProbe.ReadMetadata(header), GgufHeaderProbe.ReadTensorTable(header)!);

        Assert.True(check.IsComplete);
        Assert.False(check.IsAccepted);
        Assert.Contains("n-gram table", Assert.Single(check.Problems), StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("mtp-Qwen3.8-27B-Q4_0")]
    [InlineData("mtp-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0")]
    [InlineData("mtp-Qwen3.8-Flash-Next-draft")]
    public void MtpOnlyHeader_IsADraftModel(string name)
    {
        var header = File.ReadAllBytes(Fixture(name));

        var check = GgufSupport.Validate(GgufHeaderProbe.ReadMetadata(header), GgufHeaderProbe.ReadTensorTable(header)!);

        Assert.True(check.IsDraftModel);
        Assert.False(check.IsAccepted);
        Assert.Contains("multi-token-prediction", Assert.Single(check.Problems), StringComparison.Ordinal);
        Assert.False(GgufSupport.Validate(GgufHeaderProbe.ReadMetadata(File.ReadAllBytes(Fixture("Qwen3.8-Flash-Next-UD-IQ1_M"))),
            GgufHeaderProbe.ReadTensorTable(File.ReadAllBytes(Fixture("Qwen3.8-Flash-Next-UD-IQ1_M")))!).IsDraftModel);
    }

    /// <summary>
    /// The DFlash drafters of the sweep (llama.cpp's <c>dflash</c> architecture): each is reported as a draft model, as the importer
    /// treats it. Unsloth's Muse-Glimmer drafter (plain DFlash) and prism-ml's Ternary-Bonsai DFlash2 drafter (convolution, selector
    /// lattice, YaRN) map completely; the Nemotron 3.5 Bolt and Super-VL drafters carry a LiLiCorr correction head, which is named.
    /// </summary>
    [Theory]
    [InlineData("dflash-kquant-Muse-Glimmer-30B", false)]
    [InlineData("Ternary-Bonsai-2-9B-DFlash2-Q8_0", false)]
    [InlineData("dflash-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-NVFP4", true)]
    [InlineData("lilicorr-NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4", true)]
    public void DflashHeader_IsADraftModel(string name, bool lilicorr)
    {
        var header = File.ReadAllBytes(Fixture(name));

        var check = GgufSupport.Validate(GgufHeaderProbe.ReadMetadata(header), GgufHeaderProbe.ReadTensorTable(header)!);

        Assert.True(check.IsComplete);
        Assert.True(check.IsDraftModel);
        Assert.False(check.IsAccepted);
        if (lilicorr)
        {
            Assert.Contains(check.Problems, p => p.Contains("LiLiCorr", StringComparison.Ordinal));
        }
        else
        {
            Assert.Contains("speculative-decoding drafter", Assert.Single(check.Problems), StringComparison.Ordinal);
        }
    }

    /// <summary>The importer maps the plain DFlash drafter's header (a sparse file of its size) to the canonical drafter.</summary>
    [Fact]
    public void DflashHeader_ImportsFromItsSparseFile()
    {
        var header = File.ReadAllBytes(Fixture("dflash-kquant-Muse-Glimmer-30B"));
        string path = Path.Combine(Path.GetTempPath(), $"sharplight-dflash-header-{Guid.NewGuid():N}.gguf");
        try
        {
            using (var file = File.Create(path))
            {
                file.Write(header);
                file.SetLength(4L << 30);
            }

            var source = GgufImporter.Open(path);
            try
            {
                Assert.Equal("dflash", source.Architecture);
                Assert.Equal([2, 14, 26, 38, 50], source.Arch<int[]>("target_layers"));
                Assert.Equal(201818, source.Arch<int>("mask_token_id"));
                Assert.Contains("fc.weight", source.Weights.Keys);
                Assert.Contains("layers.4.mlp.down.weight", source.Weights.Keys);
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

    /// <summary>
    /// ISTA-DASLab's DeepSeek-V4.1-Flash GSQ-RCO uploads lack their last shard: a split model whose shards hold fewer tensors than
    /// its <c>split.tensors.count</c> is reported as incomplete instead of accepted (the importer fails on the missing file).
    /// </summary>
    [Fact]
    public void SplitHeaderWithoutAShard_IsRejected()
    {
        var header = File.ReadAllBytes(Fixture("DeepSeek-V4.1-Flash-1017-UD-Q2_K_XL"));
        var metadata = GgufHeaderProbe.ReadMetadata(header);
        var table = GgufHeaderProbe.ReadTensorTable(header)!;
        Assert.True(Convert.ToInt32(metadata["split.count"]) > 1);

        var check = GgufSupport.Validate(metadata, table.Take(table.Count - 10).ToList());

        Assert.False(check.IsAccepted);
        Assert.Contains("a shard is missing", Assert.Single(check.Problems), StringComparison.Ordinal);
        Assert.True(GgufSupport.Validate(metadata, table).IsAccepted);
    }
}
