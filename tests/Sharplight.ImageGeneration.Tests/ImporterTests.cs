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
        Assert.Same(importer.GetType(), (TestHost.Loader.Importers.TryFind(singleFile, out var found) ? found.GetType() : null));
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

            foreach (string key in new[] { "text_encoder.activation", "text_encoder.layer_count", "text_encoder.clip_skip", "text_encoder.final_norm", "scheduler.prediction_type" })
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

    public static TheoryData<string> SingleFiles => ["single", "ckpt", "f16", "Q8_0", "Q4_0", "sd21", "sdxl", "sdxl-Q8_0", "sdxl-Q4_0", "sdxl-Q4_K", "sdxl-turbo"];

    [Theory]
    [MemberData(nameof(SingleFiles))]
    public void SingleFileCheckpointsMapToTheDiffusersNames(string variant)
    {
        string path = variant switch
        {
            "single" => Env.Sd15SingleFile,
            "ckpt" => Env.Sd15Ckpt,
            "sd21" => Env.Sd21SingleFile,
            "sdxl" => Env.SdxlSingleFile,
            _ when variant.StartsWith("sdxl-", StringComparison.Ordinal) => Env.SdxlGguf(variant["sdxl-".Length..]),
            _ => Env.Sd15Gguf(variant),
        };
        string diffusers = variant switch
        {
            "sd21" => Env.Sd21Diffusers,
            _ when variant.StartsWith("sdxl", StringComparison.Ordinal) => Env.SdxlDiffusers,
            _ => Env.Sd15Diffusers,
        };
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
            // SDXL-Turbo is the SDXL network trained for 512×512.
            Assert.Equal(variant == "sdxl-turbo" ? 64L : Convert.ToInt64(folder.Metadata["latent_diffusion.unet.sample_size"]),
                Convert.ToInt64(file.Metadata["latent_diffusion.unet.sample_size"]));
            foreach (string key in new[] { "unet.head_counts", "text_encoder.head_count", "text_encoder.activation", "text_encoder.clip_skip",
                         "scheduler.prediction_type", "vae.scaling_factor", "zero_empty_negative" })
            {
                Assert.Equal(folder.Metadata["latent_diffusion." + key], file.Metadata["latent_diffusion." + key]);
            }

            if (variant.StartsWith("sdxl", StringComparison.Ordinal))
            {
                foreach (string key in new[] { "text_encoder_2.layer_count", "text_encoder_2.head_count", "text_encoder_2.activation", "text_encoder_2.clip_skip",
                             "text_encoder_2.final_norm", "text_encoder_2.pad_id", "unet.class_embedding_input", "unet.addition_time_dim",
                             "unet.down_transformer_depths", "unet.up_transformer_depths", "unet.mid_transformer_depth" })
                {
                    Assert.Equal(folder.Metadata["latent_diffusion." + key], file.Metadata["latent_diffusion." + key]);
                }
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
        Assert.Same(importer.GetType(), (TestHost.Loader.Importers.TryFind(Env.Sd15Ckpt, out var found) ? found.GetType() : null));
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

    [Theory]
    [InlineData("sd3", "joint_blocks.0.x_block.attn.qkv.weight", "Stable Diffusion 3 (diffusion transformer)")]
    [InlineData("chroma", "double_blocks.0.img_attn.qkv.weight", "chroma (diffusion transformer)")]
    [InlineData(null, "model.diffusion_model.joint_blocks.0.x_block.attn.qkv.weight", "Stable Diffusion 3 (diffusion transformer)")]
    public void DiffusionTransformerGguf_IsClaimedAndRejectedAsNotSupportedYet(string? architecture, string tensor, string family)
    {
        string directory = Path.Combine(Path.GetTempPath(), "sharplight-dit-" + Guid.NewGuid().ToString("N")[..8]);
        Directory.CreateDirectory(directory);
        try
        {
            string path = Path.Combine(directory, "model.gguf");
            WriteGguf(path, architecture, tensor);
            var importer = new DiffusersImporter();
            Assert.True(importer.CanOpen(path));
            var error = Assert.Throws<NotSupportedException>(() => importer.Open(path));
            Assert.StartsWith($"Not supported yet: {family}.", error.Message, StringComparison.Ordinal);
        }
        finally
        {
            Directory.Delete(directory, true);
        }
    }

    /// <summary>A minimal GGUF v3 file: an optional <c>general.architecture</c> and one 4-element f32 tensor.</summary>
    private static void WriteGguf(string path, string? architecture, string tensor)
    {
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8);
        void String(string text)
        {
            var bytes = System.Text.Encoding.UTF8.GetBytes(text);
            writer.Write((ulong)bytes.Length);
            writer.Write(bytes);
        }

        writer.Write("GGUF"u8);
        writer.Write(3u);
        writer.Write(1UL);
        writer.Write(architecture is null ? 0UL : 1UL);
        if (architecture is not null)
        {
            String("general.architecture");
            writer.Write(8u);
            String(architecture);
        }

        String(tensor);
        writer.Write(1u);
        writer.Write(4UL);
        writer.Write(0u);
        writer.Write(0UL);
        while (stream.Position % 32 != 0)
        {
            writer.Write((byte)0);
        }

        writer.Write(new byte[16]);
    }
}
