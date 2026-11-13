namespace Sharplight.Hub.Tests;

public sealed class SelectorTests
{
    private const string Commit = "abc";

    [Theory]
    [InlineData("Qwen_Qwen3-0.6B-Q4_K_M.gguf", "Q4_K_M")]
    [InlineData("Qwen3-0.6B-Q8_0.gguf", "Q8_0")]
    [InlineData("qwen2.5-0.5b-instruct-q4_k_m.gguf", "Q4_K_M")]
    [InlineData("Qwen3-VL-2B-Instruct-UD-Q4_K_XL.gguf", "UD-Q4_K_XL")]
    [InlineData("Qwen_Qwen3-0.6B-bf16.gguf", "BF16")]
    [InlineData("gemma-3-4b-it-f16.gguf", "F16")]
    [InlineData("Model-IQ2_XXS.gguf", "IQ2_XXS")]
    [InlineData("Q4_K_M/Qwen3-235B-A22B-Q4_K_M-00001-of-00003.gguf", "Q4_K_M")]
    [InlineData("gpt-oss-20b-MXFP4.gguf", "MXFP4")]
    [InlineData("Qwen3-0.6B.gguf", null)]
    public void ReadsQuantizationFromFileName(string path, string? label)
    {
        Assert.Equal(label, GgufFileName.QuantLabel(path));
    }

    [Fact]
    public void DefaultsToQ4KMThenQ8()
    {
        var both = Repository("a-Q8_0.gguf", "a-Q4_K_M.gguf", "a-Q2_K.gguf", "a-F16.gguf");
        var noQ4 = Repository("a-Q8_0.gguf", "a-Q2_K.gguf", "a-F16.gguf");

        Assert.Equal("Q4_K_M", ModelFileSelector.Select(both, null).Tag);
        Assert.Equal("Q8_0", ModelFileSelector.Select(noQ4, null).Tag);
    }

    [Fact]
    public void FallsBackToSmallestUnknownVariant()
    {
        var repository = new HubRepository("o/r", "main", Commit, false, [new HubFile("big-x.gguf", 100, null, null), new HubFile("small-y.gguf", 10, null, null)]);

        var selection = ModelFileSelector.Select(repository, null);

        Assert.Equal("small-y.gguf", Assert.Single(selection.Files).Path);
    }

    [Fact]
    public void SelectsTagCaseInsensitivelyOrByFileName()
    {
        var repository = Repository("m-Q8_0.gguf", "m-Q4_K_M.gguf");

        Assert.Equal("m-Q8_0.gguf", ModelFileSelector.Select(repository, "q8_0").PrimaryPath);
        Assert.Equal("m-Q8_0.gguf", ModelFileSelector.Select(repository, "m-Q8_0.gguf").PrimaryPath);
        Assert.Equal("Q8_0", ModelFileSelector.Select(repository, "m-Q8_0").Tag);
    }

    [Fact]
    public void UnknownTagListsAvailableVariants()
    {
        var repository = Repository("m-Q8_0.gguf", "m-Q4_K_M.gguf");

        var error = Assert.Throws<HubException>(() => ModelFileSelector.Select(repository, "Q5_K_S"));

        Assert.Contains("Q4_K_M, Q8_0", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void SplitModelSelectsEveryShardAndTheFirstAsPrimary()
    {
        var repository = Repository(
            "Q4_K_M/M-Q4_K_M-00002-of-00003.gguf",
            "Q4_K_M/M-Q4_K_M-00001-of-00003.gguf",
            "Q4_K_M/M-Q4_K_M-00003-of-00003.gguf",
            "Q8_0/M-Q8_0-00001-of-00002.gguf",
            "Q8_0/M-Q8_0-00002-of-00002.gguf");

        var selection = ModelFileSelector.Select(repository, null);

        Assert.Equal("Q4_K_M", selection.Tag);
        Assert.Equal(3, selection.Files.Count);
        Assert.Equal("Q4_K_M/M-Q4_K_M-00001-of-00003.gguf", selection.PrimaryPath);
        Assert.Equal(["Q4_K_M", "Q8_0"], ModelFileSelector.ListVariants(repository.Files.Select(file => file.Path)));
    }

    [Fact]
    public void IncompleteSplitModelIsRejected()
    {
        var repository = Repository("M-Q8_0-00001-of-00003.gguf", "M-Q8_0-00003-of-00003.gguf");

        var error = Assert.Throws<HubException>(() => ModelFileSelector.Select(repository, null));

        Assert.Contains("split into 3 files", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AddsVisionProjectorPreferringF16()
    {
        var repository = Repository("vl-Q4_K_M.gguf", "vl-Q8_0.gguf", "mmproj-F32.gguf", "mmproj-F16.gguf", "mmproj-BF16.gguf");

        var selection = ModelFileSelector.Select(repository, null);

        Assert.Equal("mmproj-F16.gguf", selection.ProjectorPath);
        Assert.Equal(["vl-Q4_K_M.gguf", "mmproj-F16.gguf"], selection.Files.Select(file => file.Path));
        Assert.DoesNotContain("F16", ModelFileSelector.ListVariants(repository.Files.Select(file => file.Path)));
    }

    [Fact]
    public void CheckpointTakesMetadataAndIndexedShards()
    {
        var repository = Repository(
            "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "README.md",
            "model.safetensors.index.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
            "consolidated.safetensors", "pytorch_model.bin", "1_Pooling/config.json", "original/params.json");
        const string Index = """{"metadata":{},"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors","c":"model-00001-of-00002.safetensors"}}""";

        Assert.Equal("model.safetensors.index.json", ModelFileSelector.IndexPath(repository, null));
        var selection = ModelFileSelector.Select(repository, null, Index);

        Assert.Equal(ModelFormat.Safetensors, selection.Format);
        Assert.Equal("", selection.Tag);
        Assert.Equal("", selection.PrimaryPath);
        Assert.Equal(
            ["1_Pooling/config.json", "chat_template.jinja", "config.json", "generation_config.json", "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"],
            selection.Files.Select(file => file.Path).Order(StringComparer.Ordinal));
    }

    [Fact]
    public void SafetensorsTagPicksCheckpointOverGguf()
    {
        var repository = Repository("config.json", "model.safetensors", "model-Q8_0.gguf");

        Assert.Equal(ModelFormat.Gguf, ModelFileSelector.Select(repository, null).Format);
        Assert.Equal(ModelFormat.Safetensors, ModelFileSelector.Select(repository, ModelFileSelector.SafetensorsTag).Format);
    }

    [Fact]
    public void OnnxSelectsVariantWithExternalData()
    {
        var repository = Repository("config.json", "onnx/model.onnx", "onnx/model.onnx_data", "onnx/model_fp16.onnx", "onnx/model_fp16.onnx_data",
            "onnxruntime/cuda/model.onnx");

        var byDefault = ModelFileSelector.Select(repository, null);
        var fp16 = ModelFileSelector.Select(repository, "fp16");

        Assert.Equal(ModelFormat.Onnx, byDefault.Format);
        Assert.Equal(["onnx/model.onnx", "onnx/model.onnx_data"], byDefault.Files.Select(file => file.Path));
        Assert.Equal("onnx/model_fp16.onnx", fp16.PrimaryPath);
        Assert.Equal(2, fp16.Files.Count);
    }

    [Fact]
    public void RepositoryWithoutModelFilesIsRejected()
    {
        var error = Assert.Throws<HubException>(() => ModelFileSelector.Select(Repository("README.md", "pytorch_model.bin"), null));

        Assert.Contains("no model files", error.Message, StringComparison.Ordinal);
    }

    private static HubRepository Repository(params string[] paths)
    {
        return new HubRepository("owner/repo", "main", Commit, false, [.. paths.Select(path => new HubFile(path, 1000, null, null))]);
    }
}
