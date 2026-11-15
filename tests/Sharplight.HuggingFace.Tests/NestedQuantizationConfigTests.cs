using System.Text.Json.Nodes;
using Xunit;

namespace Sharplight.HuggingFace.Tests;

/// <summary>
/// Multimodal checkpoints that keep <c>quantization_config</c> inside <c>text_config</c> (Kimi-K3, K3.1-Pro): the support check and
/// the weight conversion must see it there as they do at the root. Each tiny quantized checkpoint (tests/Sharplight.Families.Tests/
/// tools/mk_tiny_quantized.py) is read once with its own config and once with the quantization config moved into a nested
/// <c>text_config</c>; the verdict and every converted weight's type and shape must agree. Skips when the checkpoints are absent.
/// </summary>
public sealed class NestedQuantizationConfigTests
{
    private static readonly string Home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

    [Theory]
    [InlineData("tiny-llama-ct-mxfp4a16")]
    [InlineData("tiny-llama-ct-nvfp4a16")]
    [InlineData("tiny-llama-ct-w4a16-g128")]
    [InlineData("tiny-llama-gptq-4bit-g128-actorder")]
    [InlineData("tiny-llama-awq-gemm")]
    [InlineData("tiny-llama-mlx-4bit-g64")]
    public void QuantizationConfigInTextConfig(string checkpoint)
    {
        string directory = Path.Combine(Home, "models", "hf", "quant", "tiny", checkpoint);
        Assert.SkipUnless(File.Exists(Path.Combine(directory, "config.json")), $"{directory} not present");
        string rootJson = File.ReadAllText(Path.Combine(directory, "config.json"));
        string nestedJson = Nest(rootJson);
        Assert.Equal(HfSupport.GetQuantizationSupport(rootJson), HfSupport.GetQuantizationSupport(nestedJson));

        using var model = SafetensorsCheckpoint.Open(directory);
        using var rootConfig = HfConfig.Parse(rootJson, directory);
        using var nestedConfig = HfConfig.Parse(nestedJson, directory);
        var expected = HfQuantizedWeights.Resolve(rootConfig, model.Tensors);
        var actual = HfQuantizedWeights.Resolve(nestedConfig, model.Tensors);
        Assert.Equal(expected.Keys.Order(StringComparer.Ordinal), actual.Keys.Order(StringComparer.Ordinal));
        foreach (var (name, tensor) in expected)
        {
            Assert.Equal(tensor.Type, actual[name].Type);
            Assert.Equal(tensor.Shape, actual[name].Shape);
        }
    }

    /// <summary>The config with its quantization objects moved from the root into a <c>text_config</c> holding the rest.</summary>
    private static string Nest(string json)
    {
        var root = JsonNode.Parse(json)!.AsObject();
        var text = new JsonObject();
        foreach (var (key, value) in root.ToList())
        {
            root.Remove(key);
            text[key] = value;
        }

        root["model_type"] = text["model_type"]?.DeepClone();
        root["text_config"] = text;
        return root.ToJsonString();
    }
}
