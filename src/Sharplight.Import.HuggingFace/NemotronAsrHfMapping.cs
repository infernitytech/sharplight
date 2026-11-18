using System.Text.Json;
using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Hugging Face Nemotron 3.5 ASR streaming checkpoints (<c>nemotron3_5_asr</c>: nvidia/nemotron-3.5-asr-streaming-0.6b, transformers format) →
/// canonical <c>conformer_asr</c> (<c>ConformerAsrAdapter</c>, <c>ConformerEncoder</c>) with the streaming settings: the causal subsampling stack, the causal
/// depthwise convolutions with a LayerNorm in the convolution module (<c>encoder.causal</c>, <c>encoder.conv.norm</c> "layer"), chunk-limited attention (chunks of
/// <c>default_num_lookahead_tokens + 1</c> rows, <c>sliding_window − 1</c> rows of left context), NeMo's front end without per-channel normalization, the RNN-T
/// prediction and joint networks, and the language prompt projector (the one-hot language indices of <c>processor_config.json</c>; the weights of the first
/// projector layer split into the part that sees the encoder rows and the part that sees the one-hot).
/// </summary>
internal sealed partial class NemotronAsrHfMapping : IHfArchitectureMapping
{
    /// <inheritdoc/>
    public string Architecture => "conformer_asr";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["nemotron3_5_asr"];

    /// <inheritdoc/>
    public IEnumerable<string> Validate(HfConfig config) => ParakeetHfMapping.ValidateEncoder(config, "nemotron3_5_asr");

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HuggingFaceImportOptions options)
    {
        string a = Architecture + ".";
        var result = ParakeetHfMapping.EncoderHyperParameters(c, a);
        var encoder = c.Root.GetProperty("encoder_config");
        result[a + "encoder.causal"] = true;
        result[a + "encoder.conv.norm"] = "layer";
        result[a + "encoder.attention.chunked"] = true;
        result[a + "encoder.attention.left_context"] = Int(encoder, "sliding_window", 57) - 1;
        result[a + "encoder.attention.lookahead"] = Int(encoder, "default_num_lookahead_tokens", 3);
        result[a + "audio.mel_normalize"] = "none";
        result[a + "decoder.type"] = "rnnt";
        result[a + "decoder.blank_id"] = c.Int("blank_token_id");
        result[a + "decoder.layer_count"] = c.GetInt("num_decoder_layers") ?? 2;
        result[a + "decoder.max_symbols_per_step"] = c.GetInt("max_symbols_per_step") ?? 10;
        result[a + "joint.activation"] = c.GetString("hidden_act") ?? "relu";
        result[a + "prompt.count"] = c.GetInt("num_prompts") ?? 128;
        result[a + "prompt.default_id"] = c.GetInt("default_prompt_id") ?? 0;
        result[a + "prompt.dictionary"] = PromptDictionary(c.Directory);
        return result;
    }

    [GeneratedRegex(@"^encoder\.subsampling\.(conv_in|layers\.(\d+)\.(depthwise_conv|pointwise_conv))\.(weight|bias)$")]
    private static partial Regex SubsamplingName();

    [GeneratedRegex(@"^decoder\.lstm\.(weight|bias)_(ih|hh)_l(\d+)$")]
    private static partial Regex LstmName();

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
    {
        var collector = new HfWeightCollector();
        int hidden = Int(c.Root.GetProperty("encoder_config"), "hidden_size", 1024);
        foreach (var (name, tensor) in tensors)
        {
            var info = tensor.ToWeightInfo();
            var subsampling = SubsamplingName().Match(name);
            if (subsampling.Success)
            {
                // conv_in is module 0; layer i holds its depthwise (2i + 1) and pointwise (2i + 2) convolutions.
                int index = subsampling.Groups[1].Value == "conv_in" ? 0
                    : (2 * int.Parse(subsampling.Groups[2].Value, System.Globalization.CultureInfo.InvariantCulture)) + (subsampling.Groups[3].Value == "depthwise_conv" ? 1 : 2);
                collector.Add(HfTransforms.Plain(info, $"encoder.subsampling.{index}.{subsampling.Groups[4].Value}"));
                continue;
            }

            if (ParakeetHfMapping.TryMapEncoder(name, info, collector))
            {
                continue;
            }

            switch (name)
            {
                case "encoder_projector.weight" or "encoder_projector.bias":
                    collector.Add(HfTransforms.Plain(info, "joint.encoder." + Suffix(name)));
                    continue;
                case "joint.head.weight" or "joint.head.bias":
                    collector.Add(HfTransforms.Plain(info, "joint.output." + Suffix(name)));
                    continue;
                case "decoder.embedding.weight":
                    collector.Add(info.As("decoder.embedding.weight"));
                    continue;
                case "decoder.decoder_projector.weight" or "decoder.decoder_projector.bias":
                    collector.Add(HfTransforms.Plain(info, "decoder.projection." + Suffix(name)));
                    continue;
                case "prompt_projector.linear_1.bias":
                    collector.Add(HfTransforms.Plain(info, "prompt.linear_1.hidden.bias"));
                    continue;
                case "prompt_projector.linear_2.weight" or "prompt_projector.linear_2.bias":
                    collector.Add(HfTransforms.Plain(info, "prompt.linear_2." + Suffix(name)));
                    continue;
                case "prompt_projector.linear_1.weight":
                    AddSplit(collector, info, hidden);
                    continue;
            }

            var lstm = LstmName().Match(name);
            if (lstm.Success)
            {
                string gate = lstm.Groups[2].Value == "ih" ? "input" : "hidden";
                collector.Add(HfTransforms.Plain(info, $"decoder.lstm.{lstm.Groups[3].Value}.{gate}.{lstm.Groups[1].Value}"));
                continue;
            }

            throw new NotSupportedException($"nemotron3_5_asr: tensor '{name}' has no canonical mapping.");
        }

        return collector.Weights;
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, IWeightSource> MapAttachments(HfConfig config, string directory) => ParakeetHfMapping.FilterBank(config);

    // [P, H + count] → the columns over the encoder rows [P, H] and the columns over the one-hot [P, count].
    private static void AddSplit(HfWeightCollector collector, WeightInfo info, int hidden)
    {
        long rows = info.Shape[0].Fixed;
        long width = info.Shape[1].Fixed;
        long prompts = width - hidden;
        byte[] Columns(long from, long count)
        {
            var values = HfTransforms.DecodeF32(info);
            var part = new float[rows * count];
            for (long r = 0; r < rows; r++)
            {
                Array.Copy(values, (r * width) + from, part, r * count, count);
            }

            return HfTransforms.EncodeF32(part);
        }

        collector.Add(WeightInfo.Derived("prompt.linear_1.hidden.weight", DataType.F32, new Shape(rows, hidden), () => Columns(0, hidden)));
        collector.Add(WeightInfo.Derived("prompt.linear_1.prompt.weight", DataType.F32, new Shape(rows, prompts), () => Columns(hidden, prompts)));
    }

    // "name=index" pairs of the processor's prompt dictionary.
    private static string PromptDictionary(string directory)
    {
        string path = Path.Combine(directory, "processor_config.json");
        if (!File.Exists(path))
        {
            return "";
        }

        using var document = JsonDocument.Parse(File.ReadAllBytes(path));
        if (!document.RootElement.TryGetProperty("prompt_dictionary", out var dictionary) || dictionary.ValueKind != JsonValueKind.Object)
        {
            return "";
        }

        return string.Join(',', dictionary.EnumerateObject().Select(p => $"{p.Name}={p.Value.GetInt32()}"));
    }

    private static string Suffix(string name) => name[(name.LastIndexOf('.') + 1)..];

    private static int Int(JsonElement element, string key, int fallback)
        => element.TryGetProperty(key, out var value) && value.ValueKind == JsonValueKind.Number ? value.GetInt32() : fallback;
}
