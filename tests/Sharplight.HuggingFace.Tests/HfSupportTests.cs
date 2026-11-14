using Sharplight.Models;
using Xunit;

namespace Sharplight.HuggingFace.Tests;

/// <summary>
/// <see cref="HfSupport"/> answers from the registered mappings and the quantization formats the importer reads, and the importer
/// rejects the formats it does not read with a typed error before touching the weights.
/// </summary>
public sealed class HfSupportTests : IDisposable
{
    private static readonly string s_models = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models/hf");

    private readonly string _dir = Path.Combine(Path.GetTempPath(), "sl-hfsupport-" + Guid.NewGuid().ToString("N"));

    public HfSupportTests() => Directory.CreateDirectory(_dir);

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    [Theory]
    [InlineData("llama", SupportLevel.Supported)]
    [InlineData("qwen3_5_moe", SupportLevel.Supported)]
    [InlineData("qwen4_exp", SupportLevel.Partial)]
    [InlineData("qwen4_exp_text", SupportLevel.Partial)]
    [InlineData("hrm_text", SupportLevel.Unsupported)]
    [InlineData(null, SupportLevel.Unsupported)]
    public void ArchitectureSupport_FollowsTheMappings(string? modelType, SupportLevel expected)
    {
        Assert.Equal(expected, HfSupport.GetArchitectureSupport(modelType));
        Assert.Equal(expected == SupportLevel.Partial, HfSupport.GetPartialSupportNote(modelType) is not null);
    }

    [Fact]
    public void ModelType_IsTheOneTheImporterSelectsBy()
    {
        Assert.Equal("qwen3", HfSupport.GetModelType("""{"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}"""));
        Assert.Equal("gemma3_text", HfSupport.GetModelType("""{"text_config": {"model_type": "gemma3_text"}}"""));
        Assert.Equal("bailing_moe_v2", HfSupport.GetModelType("""{"architectures": ["BailingMoeV2ForCausalLM"]}"""));
        Assert.Null(HfSupport.GetModelType("not json"));
        Assert.Null(HfSupport.GetModelType("[1, 2]"));
        Assert.Contains("llama", HfSupport.ModelTypes);
    }

    [Theory]
    [InlineData("""{"model_type": "qwen3"}""", null, SupportLevel.Supported)]
    [InlineData("""{"model_type": "qwen3", "quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]}}""", "fp8", SupportLevel.Supported)]
    [InlineData("""{"model_type": "qwen2", "quantization_config": {"bits": 4, "quant_method": "gptq"}}""", "gptq (4-bit)", SupportLevel.Supported)]
    [InlineData("""{"model_type": "qwen2", "quantization_config": {"bits": 4, "quant_method": "gptq", "checkpoint_format": "marlin"}}""", "gptq (4-bit)", SupportLevel.Unsupported)]
    [InlineData("""{"model_type": "qwen2", "quantization_config": {"bits": 4, "quant_method": "awq"}}""", "awq (gemm)", SupportLevel.Supported)]
    [InlineData("""{"model_type": "qwen2", "quantization_config": {"bits": 4, "quant_method": "awq", "version": "gemv_fast"}}""", "awq (gemv_fast)", SupportLevel.Unsupported)]
    [InlineData("""{"model_type": "llama", "quantization_config": {"quant_method": "bitsandbytes", "load_in_4bit": true}}""", "bitsandbytes", SupportLevel.Supported)]
    [InlineData("""{"model_type": "qwen3_5", "quantization_config": {"group_size": 64, "bits": 8}}""", "mlx (8-bit)", SupportLevel.Supported)]
    [InlineData("""{"model_type": "qwen3", "quantization": {"group_size": 64, "bits": 4}}""", "mlx (4-bit)", SupportLevel.Supported)]
    [InlineData("""{"model_type": "qwen3", "quantization": {"group_size": 32, "bits": 4, "mode": "mxfp4"}}""", "mlx (mxfp4)", SupportLevel.Unsupported)]
    [InlineData("""{"model_type": "gpt_oss", "quantization_config": {"quant_method": "mxfp4"}}""", "mxfp4", SupportLevel.Supported)]
    [InlineData("""{"model_type": "llama", "quantization_config": {"quant_method": "mxfp4"}}""", "mxfp4", SupportLevel.Unsupported)]
    [InlineData("""{"model_type": "llama", "quantization_config": {"quant_method": "compressed-tensors", "config_groups": {"g": {"weights": {"type": "float", "num_bits": 8}}}}}""", "compressed-tensors (float 8-bit)", SupportLevel.Supported)]
    [InlineData("""{"model_type": "llama", "quantization_config": {"quant_method": "compressed-tensors", "config_groups": {"g": {"weights": {"type": "int", "num_bits": 4}}}}}""", "compressed-tensors (int 4-bit)", SupportLevel.Supported)]
    [InlineData("""{"model_type": "llama", "quantization_config": {"quant_method": "compressed-tensors", "config_groups": {"g": {"weights": {"type": "float", "num_bits": 4}}}}}""", "compressed-tensors (float 4-bit)", SupportLevel.Supported)]
    [InlineData("""{"model_type": "llama", "quantization_config": {"quant_method": "compressed-tensors", "config_groups": {"g": {"weights": {"type": "int", "num_bits": 4}}}, "sparsity_config": {"format": "sparse-24-bitmask"}}}""", "compressed-tensors (sparse-24-bitmask)", SupportLevel.Unsupported)]
    [InlineData("""{"model_type": "llama", "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"}}""", "modelopt NVFP4", SupportLevel.Supported)]
    [InlineData("""{"model_type": "llama", "quantization_config": {"quant_method": "modelopt", "quant_algo": "W4A8_AWQ"}}""", "modelopt W4A8_AWQ", SupportLevel.Unsupported)]
    [InlineData("""{"model_type": "llama", "quantization_config": {"quant_method": "quark", "global_quant_config": {"weight": {"dtype": "fp4", "group_size": 32, "scale_format": "e8m0"}}}}""", "quark (mxfp4)", SupportLevel.Supported)]
    [InlineData("""{"model_type": "llama", "quantization_config": {"quant_method": "quark", "global_quant_config": {"weight": {"dtype": "fp8_e4m3"}}, "layer_quant_config": {"*mlp*": {"weight": {"dtype": "fp4", "group_size": 32, "scale_format": "e8m0"}}}}}""", "quark (fp8, mxfp4)", SupportLevel.Supported)]
    [InlineData("""{"model_type": "llama", "quantization_config": {"quant_method": "quark", "global_quant_config": {"weight": {"dtype": "int4", "group_size": 128}}}}""", "quark (int4)", SupportLevel.Unsupported)]
    [InlineData("""{"model_type": "llama", "quantization_config": {"quant_method": "something-new"}}""", "something-new", null)]
    public void Quantization_ReadOrNot(string config, string? method, SupportLevel? expected)
    {
        var support = HfSupport.GetQuantizationSupport(config);
        Assert.Equal(method, support.Method);
        Assert.Equal(expected, support.Level);
    }

    [Fact]
    public void Quantization_ModelOptConfigFile()
    {
        const string Config = """{"model_type": "nemotron_h"}""";
        Assert.Equal(SupportLevel.Supported, HfSupport.GetQuantizationSupport(Config, """{"quantization": {"quant_algo": "FP8"}}""").Level);
        var nvfp4 = HfSupport.GetQuantizationSupport(Config, """{"quantization": {"quant_algo": "NVFP4"}}""");
        Assert.Equal(("modelopt NVFP4", SupportLevel.Supported), (nvfp4.Method, nvfp4.Level));
    }

    [Fact]
    public void RealConfigs()
    {
        string[] names = ["tiny-random-hrm-text", "tiny-random-kimi-k3", "tiny-random-gpt-oss-mxfp4", "tiny-random-qwen3.8-flash-next", "Qwen3-0.6B"];
        Assert.SkipUnless(names.All(name => File.Exists(Path.Combine(s_models, name, "config.json"))), "test configs not found");
        string Read(string name) => File.ReadAllText(Path.Combine(s_models, name, "config.json"));

        Assert.Equal(SupportLevel.Unsupported, HfSupport.GetArchitectureSupport(HfSupport.GetModelType(Read("tiny-random-hrm-text"))));
        Assert.Equal(SupportLevel.Unsupported, HfSupport.GetArchitectureSupport(HfSupport.GetModelType(Read("tiny-random-kimi-k3"))));
        // Kimi-K3's experts are MXFP4 packed by compressed-tensors, which is read (its architecture is not).
        Assert.Equal(SupportLevel.Supported, HfSupport.GetQuantizationSupport(Read("tiny-random-kimi-k3")).Level);
        Assert.Equal(SupportLevel.Supported, HfSupport.GetQuantizationSupport(Read("tiny-random-gpt-oss-mxfp4")).Level);
        Assert.Equal(SupportLevel.Partial, HfSupport.GetArchitectureSupport(HfSupport.GetModelType(Read("tiny-random-qwen3.8-flash-next"))));
        Assert.Equal(SupportLevel.Supported, HfSupport.GetArchitectureSupport(HfSupport.GetModelType(Read("Qwen3-0.6B"))));
        Assert.Equal(new HfQuantizationSupport(null, SupportLevel.Supported), HfSupport.GetQuantizationSupport(Read("Qwen3-0.6B")));
    }

    [Fact]
    public void Import_UnreadQuantization_FailsWithTheMethod()
    {
        File.WriteAllText(Path.Combine(_dir, "config.json"), """{"model_type": "qwen2", "quantization_config": {"bits": 2, "quant_method": "aqlm"}}""");
        var error = Assert.Throws<UnsupportedWeightFormatException>(() => HuggingFaceImporter.Open(_dir));
        Assert.Equal("aqlm", error.WeightFormat);
        Assert.Contains("aqlm", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Import_UnknownModelType_NamesIt()
    {
        File.WriteAllText(Path.Combine(_dir, "config.json"), """{"model_type": "hrm_text", "architectures": ["HrmTextForCausalLM"]}""");
        var error = Assert.Throws<UnsupportedArchitectureException>(() => HuggingFaceImporter.Open(_dir));
        Assert.Equal("hrm_text", error.Architecture);
    }
}
