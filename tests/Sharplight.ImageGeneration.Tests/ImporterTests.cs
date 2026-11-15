using Sharplight.Import.Diffusers;
using Sharplight.Models;
using Sharplight.Tokenizers;

namespace Sharplight.ImageGeneration.Tests;

/// <summary>The diffusers / single-file / GGUF importer: canonical names, configurations and the CLIP tokenizer.</summary>
public sealed class ImporterTests
{
    [Fact]
    public void SingleFileCheckpointEqualsDiffusersFolder()
    {
        Assert.SkipUnless(Directory.Exists(Env.Tiny), "tiny-random-sd missing");
        var importer = new DiffusersImporter();
        string singleFile = Path.Combine(Env.Tiny, "tiny-sd.safetensors");
        Assert.True(importer.CanOpen(Env.TinyDiffusers));
        Assert.True(importer.CanOpen(singleFile));
        Assert.Same(importer.GetType(), ModelImporterRegistry.Find(singleFile)?.GetType());
        var folder = importer.Open(Env.TinyDiffusers);
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

    public static TheoryData<string> Sd15Files => ["single", "f16", "Q8_0", "Q4_0"];

    [Theory]
    [MemberData(nameof(Sd15Files))]
    public void Sd15CheckpointsMapToTheDiffusersNames(string variant)
    {
        string path = variant == "single" ? Env.Sd15SingleFile : Env.Sd15Gguf(variant);
        Assert.SkipUnless(File.Exists(path) && Directory.Exists(Env.Sd15Diffusers), "SD 1.5 files missing");
        var importer = new DiffusersImporter();
        var folder = importer.Open(Env.Sd15Diffusers);
        var file = importer.Open(path);
        try
        {
            Assert.Equal(folder.Weights.Keys.Order(StringComparer.Ordinal), file.Weights.Keys.Order(StringComparer.Ordinal));
            foreach (var (name, weight) in folder.Weights)
            {
                Assert.Equal(weight.Shape, file.Weights[name].Shape);
            }

            Assert.Equal(folder.Metadata["latent_diffusion.unet.head_counts"], file.Metadata["latent_diffusion.unet.head_counts"]);
            Assert.Equal(folder.Metadata["latent_diffusion.text_encoder.head_count"], file.Metadata["latent_diffusion.text_encoder.head_count"]);
        }
        finally
        {
            foreach (var resource in folder.OwnedResources.Concat(file.OwnedResources))
            {
                resource.Dispose();
            }
        }
    }
}
