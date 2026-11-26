using System.Text.Json;
using Sharplight.Models;
using Xunit;

namespace Sharplight.Import.HuggingFace.Tests;

/// <summary>
/// The activation scheme each quantized weight carries (<see cref="WeightInfo.Activations"/>) comes from the checkpoint's quantization
/// config: weight-only formats (ModelOpt <c>W4A16_NVFP4</c>, compressed-tensors groups without <c>input_activations</c>, GPTQ, torchao
/// weight-only) give <see cref="ActivationScheme.Unquantized"/>, formats that quantize activations (ModelOpt <c>FP8</c> / <c>NVFP4</c>,
/// groups with <c>input_activations</c>, <c>quant_method fp8</c>) <see cref="ActivationScheme.Quantized"/>, anything else unspecified.
/// </summary>
public sealed class ActivationSchemeImportTests : IDisposable
{
    private static readonly string Home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

    private readonly string _directory = Path.Combine(Path.GetTempPath(), "sharplight-activation-scheme-" + Guid.NewGuid().ToString("N"));

    public ActivationSchemeImportTests() => Directory.CreateDirectory(_directory);

    public void Dispose() => Directory.Delete(_directory, recursive: true);

    /// <summary>ModelOpt's exported config (Nemotron-3.5 Bolt and Super-VL): compressed-tensors groups naming their modules.</summary>
    [Fact]
    public void ConfigGroupsDecideByInputActivations()
    {
        var schemes = Schemes("""
            {"quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "config_groups": {
              "group_0": {"quant_algo": "Q8_0", "packing": "ggml", "targets": ["model.layers.1.mixer.q_proj"]},
              "group_1": {"weights": {"num_bits": 4, "type": "float", "group_size": 16}, "targets": ["model.layers.0.mixer.in_proj"]},
              "group_2": {"input_activations": {"dynamic": false, "num_bits": 8, "type": "float"},
                          "weights": {"num_bits": 8, "type": "float"}, "targets": ["model.layers.0.mixer.out_proj", "re:.*experts\\.\\d+\\.up_proj$"]}}}}
            """);
        Assert.Equal(ActivationScheme.Unquantized, schemes.For("model.layers.0.mixer.in_proj.weight"));
        Assert.Equal(ActivationScheme.Quantized, schemes.For("model.layers.0.mixer.out_proj.weight"));
        Assert.Equal(ActivationScheme.Quantized, schemes.For("model.layers.1.mixer.experts.7.up_proj.weight"));
        Assert.Equal(ActivationScheme.Unspecified, schemes.For("model.layers.1.mixer.q_proj.weight"));
        Assert.Equal(ActivationScheme.Unspecified, schemes.For("model.layers.1.mixer.experts.7.down_proj.weight"));
    }

    /// <summary>compressed-tensors with a class target: every module of the group (W8A16 FP8 here, weight-only).</summary>
    [Fact]
    public void ClassTargetsCoverEveryModule()
    {
        var schemes = Schemes("""
            {"quantization_config": {"quant_method": "compressed-tensors", "config_groups": {
              "group_0": {"weights": {"num_bits": 8, "type": "float", "strategy": "channel"}, "input_activations": null, "targets": ["Linear"]}}}}
            """);
        Assert.Equal(ActivationScheme.Unquantized, schemes.For("model.layers.3.mlp.down_proj.weight"));
    }

    /// <summary><c>hf_quant_config.json</c>: the per-layer algorithms of a mixed-precision checkpoint, the checkpoint's otherwise.</summary>
    [Fact]
    public void ModelOptAlgorithmsPerLayer()
    {
        using var producer = JsonDocument.Parse("""
            {"quant_algo": "MIXED_PRECISION", "quantized_layers": {
              "backbone.layers.0.mixer.in_proj": {"quant_algo": "W4A16_NVFP4", "group_size": 16},
              "backbone.layers.1.mixer.experts": {"quant_algo": "NVFP4", "group_size": 16},
              "backbone.layers.2.mixer.in_proj": {"quant_algo": "FP8"},
              "backbone.layers.3.mixer.in_proj": {"quant_algo": "FP8_PB_WO"}}}
            """);
        using var config = HfConfig.Parse("{}", "");
        var schemes = HfActivationSchemes.From(config, producer.RootElement);
        Assert.Equal(ActivationScheme.Unquantized, schemes.For("backbone.layers.0.mixer.in_proj.weight"));
        Assert.Equal(ActivationScheme.Quantized, schemes.For("backbone.layers.1.mixer.experts.12.down_proj.weight"));
        Assert.Equal(ActivationScheme.Quantized, schemes.For("backbone.layers.2.mixer.in_proj.weight"));
        Assert.Equal(ActivationScheme.Unquantized, schemes.For("backbone.layers.3.mixer.in_proj.weight"));
        Assert.Equal(ActivationScheme.Unspecified, schemes.For("backbone.layers.4.mixer.in_proj.weight"));
    }

    [Theory]
    [InlineData("""{"quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [128, 128]}""", ActivationScheme.Quantized)]
    [InlineData("""{"quant_method": "modelopt", "quant_algo": "NVFP4"}""", ActivationScheme.Quantized)]
    [InlineData("""{"quant_method": "modelopt", "quant_algo": "W4A16_AWQ"}""", ActivationScheme.Unquantized)]
    [InlineData("""{"quant_method": "gptq", "bits": 4}""", ActivationScheme.Unquantized)]
    [InlineData("""{"quant_method": "torchao", "quant_type": {"default": {"_type": "Float8WeightOnlyConfig"}}}""", ActivationScheme.Unquantized)]
    [InlineData("""{"quant_method": "torchao", "quant_type": {"default": {"_type": "Float8DynamicActivationFloat8WeightConfig"}}}""", ActivationScheme.Quantized)]
    [InlineData("""{"quant_method": "quanto", "weights": "float8", "activations": null}""", ActivationScheme.Unquantized)]
    [InlineData("""{"quant_method": "bitsandbytes", "load_in_8bit": true}""", ActivationScheme.Unspecified)]
    public void CheckpointWideMethods(string quantizationConfig, ActivationScheme expected)
        => Assert.Equal(expected, Schemes($$"""{"quantization_config": {{quantizationConfig}}}""").For("model.layers.0.self_attn.q_proj.weight"));

    /// <summary>The converted FP8 weights of a weight-only compressed-tensors checkpoint carry the scheme; a W8A8 one's the other.</summary>
    [Theory]
    [InlineData(false, ActivationScheme.Unquantized)]
    [InlineData(true, ActivationScheme.Quantized)]
    public void ConvertedWeightsCarryTheScheme(bool inputActivations, ActivationScheme expected)
    {
        const int Rows = 128, Columns = 128;
        new SafetensorsWriter()
            .Add("model.layers.0.self_attn.q_proj.weight", "F8_E4M3", [Rows, Columns], new byte[Rows * Columns])
            .AddF32("model.layers.0.self_attn.q_proj.weight_scale", [Rows, 1], [.. Enumerable.Repeat(0.5f, Rows)])
            .Write(Path.Combine(_directory, "model.safetensors"));
        string input = inputActivations ? """ "input_activations": {"num_bits": 8, "type": "float", "dynamic": true},""" : "";
        string group = $"{input} \"weights\": {{\"num_bits\": 8, \"type\": \"float\", \"strategy\": \"channel\"}}, \"targets\": [\"Linear\"]";
        File.WriteAllText(Path.Combine(_directory, "config.json"),
            $"{{\"model_type\": \"llama\", \"quantization_config\": {{\"quant_method\": \"compressed-tensors\", \"config_groups\": {{\"group_0\": {{{group}}}}}}}}}");

        using var config = HfConfig.Load(_directory);
        using var model = SafetensorsCheckpoint.Open(_directory);
        var weight = HfQuantizedWeights.Resolve(config, model.Tensors)["model.layers.0.self_attn.q_proj.weight"].ToWeightInfo();
        Assert.Equal(expected, weight.Activations);
    }

    /// <summary>
    /// The Nemotron-3.5 Bolt and Super-VL release cuts of <c>LogprobReferenceTests</c> import with their declared schemes: Bolt's NVFP4
    /// Mamba projections weight-only, Super-VL's FP8 projections and NVFP4 experts with quantized activations (skips without the cuts).
    /// </summary>
    [Theory]
    [InlineData("Nemotron-3.5-Bolt-74B-A9B-24GB-L8", ActivationScheme.Unquantized)]
    [InlineData("Nemotron-3.5-Super-VL-120B-A12B-NVFP4-L8", ActivationScheme.Quantized)]
    public void ReleaseCutsImportTheirSchemes(string cut, ActivationScheme mamba)
    {
        string directory = Path.Combine(Home, "models", "hf", cut);
        Assert.SkipUnless(File.Exists(Path.Combine(directory, "config.json")), $"{directory} not present");
        var source = HuggingFaceImporter.Open(directory);
        try
        {
            var projections = source.Weights.Values
                .Where(w => w.Name.StartsWith("layers.", StringComparison.Ordinal) && w.Name.EndsWith(".ssm.in.weight", StringComparison.Ordinal)).ToList();
            Assert.NotEmpty(projections);
            Assert.All(projections, w => Assert.Equal(mamba, w.Activations));
        }
        finally
        {
            foreach (var resource in source.OwnedResources)
            {
                resource.Dispose();
            }
        }
    }

    private static HfActivationSchemes Schemes(string json)
    {
        using var config = HfConfig.Parse(json, "");
        return HfActivationSchemes.From(config, null);
    }
}
