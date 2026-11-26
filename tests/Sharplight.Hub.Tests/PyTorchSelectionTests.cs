namespace Sharplight.Hub.Tests;

/// <summary>PyTorch checkpoints on the Hub: format detection in search results and details, and the files a pull takes.</summary>
public sealed class PyTorchSelectionTests
{
    private const string Commit = "abc";

    /// <summary>ByteDance-Seed/SeedVR2-3B as listed by the Hub (sizes in bytes).</summary>
    private static readonly (string Path, long Size)[] s_seedVr3b =
    [
        (".gitattributes", 1_707), ("README.md", 5_250), ("assets/teaser.png", 900_000),
        ("apex-0.1-cp310-cp310-linux_x86_64.whl", 4_857_175), ("apex-0.1-cp39-cp39-linux_x86_64.whl", 4_839_299),
        ("ema_vae.pth", 1_002_691_902), ("neg_emb.pt", 656_540), ("pos_emb.pt", 595_100), ("seedvr2_ema_3b.pth", 13_566_090_228),
    ];

    /// <summary>ByteDance-Seed/SeedVR2-7B: two alternative DiTs (standard and sharp) and the shared VAE.</summary>
    private static readonly (string Path, long Size)[] s_seedVr7b =
    [
        (".gitattributes", 1_561), ("README.md", 5_247), ("ema_vae.pth", 1_002_691_902),
        ("seedvr2_ema_7b.pth", 32_958_774_606), ("seedvr2_ema_7b_sharp.pth", 32_958_774_606),
    ];

    private static HubRepository Repo(params (string Path, long Size)[] files)
        => new("o/r", "main", Commit, false, [.. files.Select(file => new HubFile(file.Path, file.Size, null, null))]);

    private static List<string> Paths(ModelSelection selection) => [.. selection.Files.Select(file => file.Path)];

    [Fact]
    public void SeedVr3b_IsAPyTorchFolder_WithItsCompanions_AndNoWheelsOrImages()
    {
        var repository = Repo(s_seedVr3b);

        Assert.Equal(ModelFormat.PyTorch, ModelFileSelector.DetectFormat(repository.Files.Select(file => file.Path)));
        Assert.Empty(ModelFileSelector.ListVariants(repository, null));
        Assert.Null(ModelFileSelector.IndexPath(repository, null));
        var selection = ModelFileSelector.Select(repository, null);

        Assert.Equal(ModelFormat.PyTorch, selection.Format);
        Assert.Equal("", selection.Tag);
        Assert.Equal("", selection.PrimaryPath);
        Assert.Equal(["ema_vae.pth", "neg_emb.pt", "pos_emb.pt", "seedvr2_ema_3b.pth"], Paths(selection));
    }

    [Fact]
    public void SeedVr7b_OffersEachDiT_WithTheSharedVae()
    {
        var repository = Repo(s_seedVr7b);

        Assert.Equal(["seedvr2_ema_7b", "seedvr2_ema_7b_sharp"], ModelFileSelector.ListVariants(repository, null));
        var standard = ModelFileSelector.Select(repository, null);
        var sharp = ModelFileSelector.Select(repository, "seedvr2_ema_7b_sharp");

        Assert.Equal("seedvr2_ema_7b", standard.Tag);
        Assert.Equal(["ema_vae.pth", "seedvr2_ema_7b.pth"], Paths(standard));
        Assert.Equal(["ema_vae.pth", "seedvr2_ema_7b_sharp.pth"], Paths(sharp));
        Assert.Equal("seedvr2_ema_7b_sharp", ModelFileSelector.Select(repository, "seedvr2_ema_7b_sharp.pth").Tag);
        var error = Assert.Throws<HubException>(() => ModelFileSelector.Select(repository, "seedvr2_ema_3b"));
        Assert.Contains("seedvr2_ema_7b, seedvr2_ema_7b_sharp", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ShardedPytorchModelBin_TakesTheIndexedShards_ConfigAndTokenizer_ButNotTrainingState()
    {
        var repository = Repo(
            ("config.json", 700), ("generation_config.json", 100), ("tokenizer.json", 9_000), ("tokenizer_config.json", 300),
            ("special_tokens_map.json", 100), ("vocab.txt", 2_000), ("README.md", 3_000),
            ("pytorch_model.bin.index.json", 20_000), ("pytorch_model-00001-of-00002.bin", 9_000_000), ("pytorch_model-00002-of-00002.bin", 4_000_000),
            ("training_args.bin", 4_000), ("optimizer.pt", 18_000_000), ("rng_state.pth", 14_000), ("scheduler.pt", 600),
            ("checkpoint-500/pytorch_model.bin", 13_000_000), ("tf_model.h5", 13_000_000), ("flax_model.msgpack", 13_000_000));
        const string index = """{"metadata": {}, "weight_map": {"a": "pytorch_model-00001-of-00002.bin", "b": "pytorch_model-00002-of-00002.bin"}}""";

        Assert.Equal(ModelFormat.PyTorch, ModelFileSelector.DetectFormat(repository.Files.Select(file => file.Path)));
        Assert.Equal("pytorch_model.bin.index.json", ModelFileSelector.IndexPath(repository, null));
        Assert.Empty(ModelFileSelector.ListVariants(repository, index));
        var selection = ModelFileSelector.Select(repository, null, index);

        Assert.Equal("", selection.PrimaryPath);
        Assert.Equal(
            ["config.json", "generation_config.json", "pytorch_model-00001-of-00002.bin", "pytorch_model-00002-of-00002.bin", "pytorch_model.bin.index.json",
             "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt"],
            Paths(selection));
    }

    [Fact]
    public void SinglePytorchModelBin_WithoutIndex()
    {
        var repository = Repo(("config.json", 700), ("preprocessor_config.json", 300), ("pytorch_model.bin", 110_000_000), ("tf_model.h5", 110_000_000));

        Assert.Equal(["config.json", "preprocessor_config.json", "pytorch_model.bin"], Paths(ModelFileSelector.Select(repository, null)));
    }

    [Fact]
    public void SingleCkpt_IsOpenedAsTheFile()
    {
        var repository = Repo(("README.md", 100), ("v1-inference.png", 50_000), ("sd-v1-4.ckpt", 4_265_380_512));

        var selection = ModelFileSelector.Select(repository, null);

        Assert.Equal(ModelFormat.PyTorch, selection.Format);
        Assert.Equal("sd-v1-4.ckpt", selection.PrimaryPath);
        Assert.Equal(["sd-v1-4.ckpt"], Paths(selection));
    }

    [Fact]
    public void AlternativeSingleFiles_AreVariants_TheShortestNameFirst()
    {
        var repository = Repo(("sd-v1-4.ckpt", 4_265_380_512), ("sd-v1-4-full-ema.ckpt", 7_703_807_346));

        Assert.Equal(["sd-v1-4", "sd-v1-4-full-ema"], ModelFileSelector.ListVariants(repository, null));
        Assert.Equal("sd-v1-4-full-ema.ckpt", ModelFileSelector.Select(repository, "sd-v1-4-full-ema").PrimaryPath);
    }

    [Fact]
    public void SafetensorsWin_WhenARepositoryHasBoth()
    {
        var repository = Repo(("config.json", 700), ("tokenizer.json", 9_000), ("model.safetensors", 500_000_000), ("pytorch_model.bin", 500_000_000), ("optimizer.pt", 100));

        var selection = ModelFileSelector.Select(repository, null);

        Assert.Equal(ModelFormat.Safetensors, selection.Format);
        Assert.DoesNotContain(Paths(selection), path => path.EndsWith(".bin", StringComparison.Ordinal) || path.EndsWith(".pt", StringComparison.Ordinal));
    }

    [Theory]
    [InlineData("optimizer.pt")]
    [InlineData("scheduler.pt")]
    [InlineData("rng_state_3.pth")]
    [InlineData("training_args.bin")]
    [InlineData("checkpoint-157/model.pt")]
    [InlineData("ggml-base.en.bin")]
    [InlineData("model.bin")]
    [InlineData("openvino/openvino_model.bin")]
    [InlineData("Encoder.mlmodelc/weights/weight.bin")]
    [InlineData("apex-0.1-cp310-cp310-linux_x86_64.whl")]
    public void JunkAndOtherRuntimes_AreNotPyTorchWeights(string path)
    {
        Assert.False(PyTorchSelection.HasWeights([path, "README.md"]));
    }

    [Theory]
    [InlineData("model.pth")]
    [InlineData("weights/best.pt")]
    [InlineData("v3_sd15_mm.ckpt")]
    [InlineData("pytorch_model.fp16.bin")]
    [InlineData("text_encoder/pytorch_model.bin")]
    public void CheckpointFiles_ArePyTorchWeights(string path)
    {
        Assert.True(PyTorchSelection.HasWeights([path]));
    }

    [Fact]
    public void BinWithAnIndex_IsAShard()
    {
        Assert.True(PyTorchSelection.HasWeights(["model-00001-of-00002.bin", "model.bin.index.json"]));
    }

    [Fact]
    public void Search_ListsPyTorchRepositories_AndNamesUnsupportedFormats()
    {
        const string json = """
            [
              {"id":"ByteDance-Seed/SeedVR2-3B","library_name":"seedvr","pipeline_tag":"video-to-video",
               "siblings":[{"rfilename":"README.md"},{"rfilename":"apex-0.1-cp39-cp39-linux_x86_64.whl"},{"rfilename":"ema_vae.pth"},
                           {"rfilename":"neg_emb.pt"},{"rfilename":"pos_emb.pt"},{"rfilename":"seedvr2_ema_3b.pth"}]},
              {"id":"facebook/convnext-tiny-224","siblings":[{"rfilename":"config.json"},{"rfilename":"pytorch_model.bin"},{"rfilename":"tf_model.h5"}]},
              {"id":"o/both","siblings":[{"rfilename":"config.json"},{"rfilename":"model.safetensors"},{"rfilename":"pytorch_model.bin"}]},
              {"id":"nvidia/parakeet","siblings":[{"rfilename":"parakeet-tdt-0.6b-v2.nemo"},{"rfilename":"README.md"}]},
              {"id":"Systran/faster-whisper-small","siblings":[{"rfilename":"config.json"},{"rfilename":"model.bin"},{"rfilename":"tokenizer.json"},{"rfilename":"vocabulary.txt"}]},
              {"id":"o/training-only","siblings":[{"rfilename":"optimizer.pt"},{"rfilename":"training_args.bin"}]}
            ]
            """;

        var results = HubJson.ParseSearch(json).ToDictionary(result => result.Id);

        Assert.Equal([ModelFormat.PyTorch], results["ByteDance-Seed/SeedVR2-3B"].Formats);
        Assert.Equal([ModelFormat.PyTorch], results["facebook/convnext-tiny-224"].Formats);
        Assert.Equal([ModelFormat.Safetensors], results["o/both"].Formats);
        Assert.Null(results["o/both"].UnsupportedFormat);
        Assert.Empty(results["nvidia/parakeet"].Formats);
        Assert.Equal("NeMo archives (.nemo)", results["nvidia/parakeet"].UnsupportedFormat);
        Assert.Equal("CTranslate2 weights (model.bin)", results["Systran/faster-whisper-small"].UnsupportedFormat);
        Assert.Empty(results["o/training-only"].Formats);
    }

    [Fact]
    public void NoSupportedFiles_SaysWhatTheRepositoryHas()
    {
        var repository = Repo(("canary-1b.nemo", 4_000_000_000), ("README.md", 100));

        var error = Assert.Throws<HubException>(() => ModelFileSelector.Select(repository, null));

        Assert.Contains("NeMo archives (.nemo)", error.Message, StringComparison.Ordinal);
    }
}
