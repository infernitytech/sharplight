using Sharplight.Import.Diffusers;
using Sharplight.Models;
using Sharplight.Tokenizers;

namespace Sharplight.ImageGeneration.Tests;

/// <summary>The diffusers / single-file / GGUF importer: canonical names, configurations and the CLIP tokenizer.</summary>
public sealed class ImporterTests
{
    [Theory]
    [InlineData("sd1", "tiny-sd.safetensors")]
    [InlineData("sd1", "tiny-sd.ckpt")]
    [InlineData("sd2", "tiny-sd2.safetensors")]
    [InlineData("sd2", "tiny-sd2.ckpt")]
    public void SingleFileCheckpointEqualsDiffusersFolder(string family, string fileName)
    {
        string singleFile = Path.Combine(Env.TinyRoot(family), fileName);
        Assert.SkipUnless(File.Exists(singleFile), $"{fileName} missing");
        var importer = new DiffusersImporter();
        Assert.True(importer.CanOpen(Env.TinyDiffusersOf(family)));
        Assert.True(importer.CanOpen(singleFile));
        Assert.Same(importer.GetType(), ModelImporterRegistry.Find(singleFile)?.GetType());
        var folder = importer.Open(Env.TinyDiffusersOf(family));
        var file = importer.Open(singleFile);
        try
        {
            Assert.Equal(folder.Weights.Keys.Order(StringComparer.Ordinal), file.Weights.Keys.Order(StringComparer.Ordinal));
            foreach (var (name, weight) in folder.Weights)
            {
                var other = file.Weights[name];
                Assert.Equal(weight.Shape, other.Shape);
                Assert.True(weight.ReadAll().AsSpan().SequenceEqual(other.ReadAll()), $"{name} differs");
            }

            foreach (string key in new[] { "block_channels", "down_transformer_depths", "up_transformer_depths", "layers_per_block", "context_dim" })
            {
                Assert.Equal(folder.Metadata["latent_diffusion.unet." + key], file.Metadata["latent_diffusion.unet." + key]);
            }

            foreach (string key in new[] { "text_encoder.activation", "text_encoder.layer_count", "scheduler.prediction_type" })
            {
                Assert.Equal(folder.Metadata["latent_diffusion." + key], file.Metadata["latent_diffusion." + key]);
            }

            Assert.Equal(folder.Metadata[CanonicalKeys.TokenizerPadId], file.Metadata[CanonicalKeys.TokenizerPadId]);
        }
        finally
        {
            foreach (var resource in folder.OwnedResources.Concat(file.OwnedResources))
            {
                resource.Dispose();
            }
        }
    }

    [Fact]
    public void ClipTokenizerMatchesTransformers()
    {
        Assert.SkipUnless(Directory.Exists(Env.Tiny), "tiny-random-sd missing");
        var source = new DiffusersImporter().Open(Path.Combine(Env.Tiny, "tiny-sd.safetensors"));
        try
        {
            // The single file has no tokenizer: the vocabulary is rebuilt from the embedded OpenAI merges.
            var tokenizer = TokenizerFactory.Create(TokenizerData.FromMetadata(source.Metadata));
            foreach (var prompt in Env.TinyIndex().GetProperty("tokens").EnumerateObject())
            {
                var expected = prompt.Value.EnumerateArray().Select(t => t.GetInt32()).TakeWhile(t => t != 49407).Append(49407).ToArray();
                Assert.Equal(expected, tokenizer.Encode(prompt.Name, addSpecial: true, parseSpecial: false));
            }
        }
        finally
        {
            foreach (var resource in source.OwnedResources)
            {
                resource.Dispose();
            }
        }
    }

    public static TheoryData<string> Sd15Files => ["single", "ckpt", "f16", "Q8_0", "Q4_0", "sd21"];

    [Theory]
    [MemberData(nameof(Sd15Files))]
    public void Sd15CheckpointsMapToTheDiffusersNames(string variant)
    {
        string path = variant switch
        {
            "single" => Env.Sd15SingleFile,
            "ckpt" => Env.Sd15Ckpt,
            "sd21" => Env.Sd21SingleFile,
            _ => Env.Sd15Gguf(variant),
        };
        string diffusers = variant == "sd21" ? Env.Sd21Diffusers : Env.Sd15Diffusers;
        Assert.SkipUnless(File.Exists(path) && Directory.Exists(diffusers), "SD files missing");
        var importer = new DiffusersImporter();
        var folder = importer.Open(diffusers);
        var file = importer.Open(path);
        try
        {
            Assert.Equal(folder.Weights.Keys.Order(StringComparer.Ordinal), file.Weights.Keys.Order(StringComparer.Ordinal));
            foreach (var (name, weight) in folder.Weights)
            {
                Assert.Equal(weight.Shape, file.Weights[name].Shape);
            }

            Assert.Equal(folder.Metadata["latent_diffusion.unet.head_counts"], file.Metadata["latent_diffusion.unet.head_counts"]);
            foreach (string key in new[] { "unet.sample_size", "text_encoder.head_count", "text_encoder.activation", "scheduler.prediction_type" })
            {
                Assert.Equal(folder.Metadata["latent_diffusion." + key], file.Metadata["latent_diffusion." + key]);
            }

            Assert.Equal(folder.Metadata[CanonicalKeys.TokenizerPadId], file.Metadata[CanonicalKeys.TokenizerPadId]);
        }
        finally
        {
            foreach (var resource in folder.OwnedResources.Concat(file.OwnedResources))
            {
                resource.Dispose();
            }
        }
    }

    [Fact]
    public void Sd15CkptHoldsTheSafetensorsWeights()
    {
        Assert.SkipUnless(File.Exists(Env.Sd15Ckpt) && File.Exists(Env.Sd15SingleFile), "SD 1.5 .ckpt / .safetensors missing");
        var importer = new DiffusersImporter();
        Assert.Same(importer.GetType(), ModelImporterRegistry.Find(Env.Sd15Ckpt)?.GetType());
        var safetensors = importer.Open(Env.Sd15SingleFile);
        var ckpt = importer.Open(Env.Sd15Ckpt);
        try
        {
            Assert.Equal("pytorch", ckpt.Format);
            foreach (string name in safetensors.Weights.Keys.Where((_, i) => i % 37 == 0))
            {
                Assert.True(safetensors.Weights[name].ReadAll().AsSpan().SequenceEqual(ckpt.Weights[name].ReadAll()), $"{name} differs");
            }
        }
        finally
        {
            foreach (var resource in safetensors.OwnedResources.Concat(ckpt.OwnedResources))
            {
                resource.Dispose();
            }
        }
    }
}
