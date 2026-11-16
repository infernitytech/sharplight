using Sharplight.Import.Snac;

namespace Sharplight.TextToSpeech.Tests;

/// <summary>The SNAC importer: the decoding half of the codec with folded weight norms.</summary>
public class SnacImporterTests
{
    [Fact]
    public void The_importer_reads_the_decoder_and_the_codebooks()
    {
        TtsModels.Require(Path.Combine(TtsModels.Snac, "pytorch_model.bin"));
        var importer = new SnacImporter();
        Assert.True(importer.CanOpen(TtsModels.Snac));
        Assert.True(importer.Priority > 0);
        var source = SnacImporter.Open(TtsModels.Snac);
        try
        {
            Assert.Equal("snac", source.Architecture);
            Assert.Equal(24000, source.Arch<int>("sample_rate"));
            Assert.Equal(new long[] { 4, 2, 1 }, source.Arch<long[]>("vq_strides"));
            Assert.Equal(768, source.Arch<int>("latent_dim"));
            Assert.DoesNotContain(source.Weights.Keys, k => k.StartsWith("encoder.", StringComparison.Ordinal) || k.Contains("parametrizations", StringComparison.Ordinal));
            Assert.Equal(new Sharplight.Tensors.Shape(4096, 8), source.Weights["quantizer.quantizers.2.codebook.weight"].Shape);
            Assert.Equal(new Sharplight.Tensors.Shape(1024, 768, 1), source.Weights["decoder.model.1.weight"].Shape);
            Assert.Equal(new Sharplight.Tensors.Shape(1024), source.Weights["decoder.model.2.block.0.alpha"].Shape);
            var folded = source.Weights["decoder.model.7.weight"].ReadAll();
            Assert.Equal(64 * 7 * sizeof(float), folded.Length);
        }
        finally
        {
            foreach (var resource in source.OwnedResources)
            {
                resource.Dispose();
            }
        }
    }
}
