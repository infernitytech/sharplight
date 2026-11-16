namespace Sharplight.Hub.Tests;

/// <summary>Download planning for Diffusers pipelines and whisper.cpp repositories, on recorded file lists.</summary>
public sealed class DiffusersSelectionTests
{
    private static HubRepository Repo(params (string Path, long Size)[] files)
        => new("o/r", "main", "abc", false, [.. files.Select(file => new HubFile(file.Path, file.Size, null, null))]);

    private static string Index(string className, params string[] components)
    {
        var parts = string.Join(",", components.Select(name => $"\"{name}\": [\"diffusers\", \"Some\"]"));
        return $"{{\"_class_name\": \"{className}\", \"_diffusers_version\": \"0.30.0\", \"safety_checker\": [null, null], {parts}}}";
    }

    private static HubRepository StableDiffusion15() => Repo(
        ("model_index.json", 541), ("README.md", 100), ("v1-5-pruned.safetensors", 7_700), ("v1-5-pruned.ckpt", 7_700),
        ("feature_extractor/preprocessor_config.json", 342),
        ("safety_checker/config.json", 4_000), ("safety_checker/model.safetensors", 1_200), ("safety_checker/pytorch_model.bin", 1_200),
        ("scheduler/scheduler_config.json", 308),
        ("text_encoder/config.json", 617), ("text_encoder/model.safetensors", 492), ("text_encoder/model.fp16.safetensors", 246),
        ("text_encoder/pytorch_model.bin", 492), ("text_encoder/pytorch_model.fp16.bin", 246),
        ("tokenizer/merges.txt", 525), ("tokenizer/special_tokens_map.json", 472), ("tokenizer/tokenizer_config.json", 806), ("tokenizer/vocab.json", 1_000),
        ("unet/config.json", 743), ("unet/diffusion_pytorch_model.safetensors", 3_438), ("unet/diffusion_pytorch_model.fp16.safetensors", 1_719),
        ("unet/diffusion_pytorch_model.non_ema.safetensors", 3_438), ("unet/diffusion_pytorch_model.bin", 3_438), ("unet/diffusion_pytorch_model.non_ema.bin", 3_438),
        ("unet/diffusion_pytorch_model.fp16.bin", 1_719), ("unet/diffusion_pytorch_model.msgpack", 3_438),
        ("vae/config.json", 551), ("vae/diffusion_pytorch_model.safetensors", 335), ("vae/diffusion_pytorch_model.fp16.safetensors", 167),
        ("vae/diffusion_pytorch_model.bin", 335));

    private static readonly string[] s_sd15Components = ["feature_extractor", "safety_checker", "scheduler", "text_encoder", "tokenizer", "unet", "vae"];

    [Fact]
    public void StableDiffusion15_OffersFp16AndFullAndDefaultsToTheSmaller()
    {
        var repo = StableDiffusion15();
        string index = Index("StableDiffusionPipeline", s_sd15Components);

        var variants = ModelFileSelector.ListVariants(repo.Files.Select(file => file.Path), index);
        var selection = ModelFileSelector.Select(repo, null, index);

        Assert.Equal(["fp16", "full"], variants);
        Assert.Equal(ModelFormat.Diffusers, selection.Format);
        Assert.Equal("fp16", selection.Tag);
        Assert.Equal("", selection.PrimaryPath);
        var paths = selection.Files.Select(file => file.Path).ToList();
        Assert.Contains("model_index.json", paths);
        Assert.Contains("unet/diffusion_pytorch_model.fp16.safetensors", paths);
        Assert.Contains("vae/diffusion_pytorch_model.fp16.safetensors", paths);
        Assert.Contains("text_encoder/model.fp16.safetensors", paths);
        Assert.Contains("tokenizer/merges.txt", paths);
        Assert.Contains("scheduler/scheduler_config.json", paths);
        Assert.Contains("feature_extractor/preprocessor_config.json", paths);
        Assert.DoesNotContain(paths, path => path.EndsWith(".bin", StringComparison.Ordinal) || path.EndsWith(".msgpack", StringComparison.Ordinal) || path.EndsWith(".ckpt", StringComparison.Ordinal));
        Assert.DoesNotContain(paths, path => path.StartsWith("safety_checker/model", StringComparison.Ordinal));
        Assert.DoesNotContain("README.md", paths);
        Assert.DoesNotContain("v1-5-pruned.safetensors", paths);
        Assert.Equal(1_719 + 167 + 246, selection.Files.Where(file => file.Path.EndsWith(".safetensors", StringComparison.Ordinal)).Sum(file => file.Size));
    }

    [Fact]
    public void StableDiffusion15_FullVariantTakesOneFullPrecisionFilePerComponent()
    {
        var repo = StableDiffusion15();

        var selection = ModelFileSelector.Select(repo, "full", Index("StableDiffusionPipeline", s_sd15Components));

        var weights = selection.Files.Where(file => file.Path.EndsWith(".safetensors", StringComparison.Ordinal)).Select(file => file.Path).Order(StringComparer.Ordinal).ToList();
        Assert.Equal(["text_encoder/model.safetensors", "unet/diffusion_pytorch_model.safetensors", "vae/diffusion_pytorch_model.safetensors"], weights);
        Assert.Equal("full", selection.Tag);
    }

    [Fact]
    public void UnknownVariant_ListsTheAvailableOnes()
    {
        var error = Assert.Throws<HubException>(() => ModelFileSelector.Select(StableDiffusion15(), "int8", Index("StableDiffusionPipeline", s_sd15Components)));

        Assert.Contains("fp16, full", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Sdxl_OneVariantWhenOnlyFullPrecisionIsOffered_AndSecondEncoderIsIncluded()
    {
        var repo = Repo(
            ("model_index.json", 600), ("scheduler/scheduler_config.json", 500),
            ("text_encoder/config.json", 600), ("text_encoder/model.safetensors", 246),
            ("text_encoder_2/config.json", 600), ("text_encoder_2/model.safetensors", 1_390),
            ("tokenizer/vocab.json", 1), ("tokenizer/merges.txt", 1), ("tokenizer_2/vocab.json", 1), ("tokenizer_2/merges.txt", 1),
            ("unet/config.json", 1), ("unet/diffusion_pytorch_model.safetensors", 5_140), ("vae/config.json", 1), ("vae/diffusion_pytorch_model.safetensors", 335),
            ("sd_xl_base_1.0.safetensors", 6_900), ("sd_xl_base_1.0_0.9vae.safetensors", 6_900));
        string index = Index("StableDiffusionXLPipeline", "scheduler", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "unet", "vae");

        var selection = ModelFileSelector.Select(repo, null, index);

        Assert.Empty(ModelFileSelector.ListVariants(repo.Files.Select(file => file.Path), index));
        Assert.Equal("", selection.Tag);
        Assert.Contains(selection.Files, file => file.Path == "text_encoder_2/model.safetensors");
        Assert.Contains(selection.Files, file => file.Path == "tokenizer_2/merges.txt");
        Assert.DoesNotContain(selection.Files, file => file.Path.StartsWith("sd_xl", StringComparison.Ordinal));
        Assert.Equal(246 + 1_390 + 5_140 + 335, selection.Files.Where(file => file.Path.EndsWith(".safetensors", StringComparison.Ordinal)).Sum(file => file.Size));
    }

    [Fact]
    public void Flux_TakesEveryShardOfTheTransformerAndItsIndex_NotTheOtherPrecision()
    {
        var repo = Repo(
            ("model_index.json", 500), ("flux1-schnell.safetensors", 23_800), ("ae.safetensors", 335),
            ("scheduler/scheduler_config.json", 200),
            ("text_encoder/config.json", 1), ("text_encoder/model.safetensors", 246),
            ("text_encoder_2/config.json", 1), ("text_encoder_2/model.safetensors.index.json", 20),
            ("text_encoder_2/model-00001-of-00002.safetensors", 4_990), ("text_encoder_2/model-00002-of-00002.safetensors", 4_530),
            ("tokenizer/vocab.json", 1), ("tokenizer/merges.txt", 1), ("tokenizer_2/spiece.model", 800), ("tokenizer_2/tokenizer.json", 2_400), ("tokenizer_2/tokenizer_config.json", 1),
            ("transformer/config.json", 1), ("transformer/diffusion_pytorch_model.safetensors.index.json", 120),
            ("transformer/diffusion_pytorch_model-00001-of-00003.safetensors", 9_950), ("transformer/diffusion_pytorch_model-00002-of-00003.safetensors", 9_950),
            ("transformer/diffusion_pytorch_model-00003-of-00003.safetensors", 3_870),
            ("vae/config.json", 1), ("vae/diffusion_pytorch_model.safetensors", 168));
        string index = Index("FluxPipeline", "scheduler", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "transformer", "vae");

        var selection = ModelFileSelector.Select(repo, null, index);

        var paths = selection.Files.Select(file => file.Path).ToList();
        Assert.Equal(3, paths.Count(path => path.StartsWith("transformer/diffusion_pytorch_model-", StringComparison.Ordinal)));
        Assert.Contains("transformer/diffusion_pytorch_model.safetensors.index.json", paths);
        Assert.Contains("text_encoder_2/model.safetensors.index.json", paths);
        Assert.Contains("tokenizer_2/spiece.model", paths);
        Assert.DoesNotContain("flux1-schnell.safetensors", paths);
        Assert.DoesNotContain("ae.safetensors", paths);
    }

    [Fact]
    public void Wan22_TakesTheFp16ShardedTransformerOfTheIndex()
    {
        var repo = Repo(
            ("model_index.json", 500), ("scheduler/scheduler_config.json", 1),
            ("text_encoder/config.json", 1), ("text_encoder/model-00001-of-00005.safetensors", 4_000), ("text_encoder/model-00005-of-00005.safetensors", 3_000),
            ("text_encoder/model.safetensors.index.json", 50),
            ("tokenizer/spiece.model", 4_000), ("tokenizer/tokenizer.json", 16_000), ("tokenizer/special_tokens_map.json", 1),
            ("transformer/config.json", 1), ("transformer/diffusion_pytorch_model-00001-of-00003.safetensors", 9_000), ("transformer/diffusion_pytorch_model-00002-of-00003.safetensors", 1_000),
            ("transformer/diffusion_pytorch_model-00003-of-00003.safetensors", 100), ("transformer/diffusion_pytorch_model.safetensors.index.json", 90),
            ("vae/config.json", 1), ("vae/diffusion_pytorch_model.safetensors", 1_400));
        string index = Index("WanPipeline", "scheduler", "text_encoder", "tokenizer", "transformer", "vae");

        var selection = ModelFileSelector.Select(repo, null, index);

        Assert.Equal(repo.Files.Count - 0, selection.Files.Count);
        Assert.Equal(repo.Files.Sum(file => file.Size), selection.TotalBytes);
    }

    [Fact]
    public void ComponentsOutsideTheIndexAreSkipped()
    {
        var repo = Repo(("model_index.json", 5), ("unet/config.json", 1), ("unet/diffusion_pytorch_model.safetensors", 10), ("vae/config.json", 1), ("vae/diffusion_pytorch_model.safetensors", 3),
            ("unet_old/diffusion_pytorch_model.safetensors", 10), ("assets/readme.json", 1));

        var selection = ModelFileSelector.Select(repo, null, Index("StableDiffusionPipeline", "unet", "vae"));

        Assert.DoesNotContain(selection.Files, file => file.Path.StartsWith("unet_old/", StringComparison.Ordinal) || file.Path.StartsWith("assets/", StringComparison.Ordinal));
        Assert.Equal(13, selection.Files.Where(file => file.Path.EndsWith(".safetensors", StringComparison.Ordinal)).Sum(file => file.Size));
    }

    [Fact]
    public void ComponentWithOnlyBinWeights_IsReported()
    {
        var repo = Repo(("model_index.json", 5), ("unet/config.json", 1), ("unet/diffusion_pytorch_model.bin", 10));

        var error = Assert.Throws<HubException>(() => ModelFileSelector.Select(repo, null, Index("StableDiffusionPipeline", "unet")));

        Assert.Contains("unet", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void PipelineIndex_ReadsClassAndSkipsNullComponents()
    {
        var pipeline = DiffusersPipeline.Parse(Index("FluxPipeline", "transformer", "vae"));

        Assert.NotNull(pipeline);
        Assert.Equal("FluxPipeline", pipeline.ClassName);
        Assert.Equal(["transformer", "vae"], pipeline.Components);
        Assert.Null(DiffusersPipeline.Parse("not json"));
    }

    [Fact]
    public void WhisperCpp_PicksOneModelFile()
    {
        var repo = Repo(("README.md", 1), ("ggml-tiny.bin", 75), ("ggml-base.bin", 142), ("ggml-base.en-q5_1.bin", 60), ("ggml-small.bin", 466), ("models/ggml-x.bin", 9));
        var paths = repo.Files.Select(file => file.Path);

        var tags = ModelFileSelector.ListVariants(paths);
        var recommended = ModelFileSelector.Select(repo, null);
        var small = ModelFileSelector.Select(repo, "small");

        Assert.Equal("base.en-q5_1", tags[0]);
        Assert.Equal(["base.en-q5_1", "base", "small", "tiny"], tags);
        Assert.Equal("ggml-base.en-q5_1.bin", recommended.PrimaryPath);
        Assert.Equal(ModelFormat.WhisperCpp, recommended.Format);
        Assert.Equal("ggml-small.bin", Assert.Single(small.Files).Path);
        Assert.Equal(466, small.TotalBytes);
        Assert.Throws<HubException>(() => ModelFileSelector.Select(repo, "large-v3"));
    }
}
