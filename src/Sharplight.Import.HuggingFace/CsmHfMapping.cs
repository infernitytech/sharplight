using System.Buffers.Binary;
using System.Text.Json;
using System.Text.Json.Nodes;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// CSM (<c>csm</c>, CsmForConditionalGeneration: a Llama backbone over text tokens and the sums of a frame's 32 codebook embeddings, a small Llama depth decoder that predicts
/// codebooks 1 to 31 of a frame, and the bundled Mimi codec) → canonical <c>csm</c> (documented on <c>CsmAdapter</c>). The backbone maps through <see cref="DecoderHfMapping"/>
/// as a Llama (its embedding rows are composed on the host; the first-codebook head stands in as the output); the depth decoder the same way with its hyper-parameters under
/// <c>csm.depth.</c> and its weights under <c>depth.</c>; the text embedding, the codebook embedding table (shared by the backbone and the depth decoder), the depth decoder's
/// embedding projector and its 31 codebook heads (read on the host) keep their names under <c>csm.</c>; the codec's weights go through <see cref="MimiHfMapping"/> under
/// <c>codec.</c> with its hyper-parameters under <c>csm.codec.</c>.
/// </summary>
internal sealed class CsmHfMapping : IHfArchitectureMapping
{
    private const string Backbone = "backbone_model.";
    private const string Depth = "depth_decoder.model.";
    private const string Codec = "codec_model.";

    /// <inheritdoc/>
    public string Architecture => "csm";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["csm"];

    /// <inheritdoc/>
    public IEnumerable<string> Validate(HfConfig c)
    {
        if (c.Root.TryGetProperty("codec_config", out var codec) && codec.TryGetProperty("model_type", out var type) && type.GetString() != "mimi")
        {
            yield return $"csm: only the Mimi codec is supported, not '{type.GetString()}'.";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HuggingFaceImportOptions options)
    {
        var md = new Dictionary<string, object>(StringComparer.Ordinal);
        var mapping = new DecoderHfMapping();
        using (var config = BackboneConfig(c))
        {
            foreach (var (key, value) in mapping.MapHyperParameters(config, options))
            {
                md[key] = value;
            }
        }

        using (var config = DepthConfig(c))
        {
            foreach (var (key, value) in mapping.MapHyperParameters(config, options))
            {
                md[Architecture + ".depth." + key] = value;
            }
        }

        var depth = c.Root.GetProperty("depth_decoder_config");
        string a = Architecture + ".";
        md[a + "text_vocab_size"] = c.Int("text_vocab_size");
        md[a + "codebook_vocab_size"] = c.Int("vocab_size");
        md[a + "codebooks"] = c.Int("num_codebooks");
        md[a + "audio_token_id"] = c.Int("audio_token_id");
        md[a + "audio_eos_token_id"] = c.Int("audio_eos_token_id");
        md[a + "codebook_eos_token_id"] = c.Int("codebook_eos_token_id");
        md[a + "codebook_pad_token_id"] = c.Int("codebook_pad_token_id");
        md[a + "bos_token_id"] = c.Int("bos_token_id");
        md[a + "depth_hidden_size"] = depth.GetProperty("hidden_size").GetInt32();
        string generation = Path.Combine(c.Directory, "generation_config.json");
        if (File.Exists(generation))
        {
            using var g = JsonDocument.Parse(File.ReadAllText(generation));
            foreach (var (key, name) in new[] { ("temperature", "temperature"), ("top_k", "top_k"), ("top_p", "top_p"), ("depth_decoder_temperature", "depth.temperature"),
                ("depth_decoder_top_k", "depth.top_k"), ("depth_decoder_top_p", "depth.top_p"), ("max_new_tokens", "max_frames") })
            {
                if (g.RootElement.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.Number)
                {
                    md[a + "sampling." + name] = (float)v.GetDouble();
                }
            }
        }

        using (var codec = CodecConfig(c))
        {
            foreach (var (key, value) in new MimiHfMapping().MapHyperParameters(codec, options))
            {
                md[a + "codec." + key["mimi.".Length..]] = value;
            }
        }

        return md;
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
    {
        var backbone = new Dictionary<string, SafetensorsTensor>(StringComparer.Ordinal);
        var depth = new Dictionary<string, SafetensorsTensor>(StringComparer.Ordinal);
        var codec = new Dictionary<string, SafetensorsTensor>(StringComparer.Ordinal);
        var collector = new HfWeightCollector();
        string a = Architecture + ".";
        SafetensorsTensor? audioEmbedding = null;
        foreach (var (name, tensor) in tensors)
        {
            if (name == "embed_text_tokens.weight")
            {
                collector.Add(HfTransforms.Plain(tensor.ToWeightInfo(), a + "text_embedding.weight"));
            }
            else if (name == "lm_head.weight")
            {
                backbone["lm_head.weight"] = tensor;
            }
            else if (name == Backbone + "embed_tokens.embed_audio_tokens.weight" || name == Depth + "embed_tokens.weight")
            {
                audioEmbedding = tensor;
            }
            else if (name == Depth + "inputs_embeds_projector.weight")
            {
                collector.Add(HfTransforms.Plain(tensor.ToWeightInfo(), a + "depth.projector.weight"));
            }
            else if (name == "depth_decoder.codebooks_head.weight")
            {
                collector.Add(HfTransforms.Plain(tensor.ToWeightInfo(), a + "depth.heads.weight"));
            }
            else if (name.StartsWith(Backbone, StringComparison.Ordinal))
            {
                backbone["model." + name[Backbone.Length..]] = tensor;
            }
            else if (name.StartsWith(Depth, StringComparison.Ordinal))
            {
                depth["model." + name[Depth.Length..]] = tensor;
            }
            else if (name.StartsWith(Codec, StringComparison.Ordinal))
            {
                codec[name[Codec.Length..]] = tensor;
            }
            else
            {
                collector.AddUnmapped(name);
            }
        }

        collector.Add(HfTransforms.Plain((audioEmbedding ?? throw new InvalidDataException("csm: the codebook embedding table is missing.")).ToWeightInfo(), a + "audio_embedding.weight"));

        // The decoders take embedding rows only; the first-codebook head stands in for the vocabulary of each (the host applies the heads).
        backbone["model.embed_tokens.weight"] = backbone["lm_head.weight"];
        var mapping = new DecoderHfMapping();
        using (var config = BackboneConfig(c))
        {
            foreach (var (_, w) in mapping.MapWeights(config, backbone, options))
            {
                collector.Add(w);
            }
        }

        var headTensor = tensors["depth_decoder.codebooks_head.weight"];
        var heads = headTensor.ToWeightInfo();
        long width = heads.Shape[1].Fixed;
        long vocabulary = heads.Shape[2].Fixed;
        depth["model.embed_tokens.weight"] = headTensor;
        depth["lm_head.weight"] = headTensor;
        using (var config = DepthConfig(c))
        {
            foreach (var (name, w) in mapping.MapWeights(config, depth, options))
            {
                if (name is "token_embedding.weight" or "lm_head.weight")
                {
                    continue;
                }

                collector.Add(w.As("depth." + name));
            }

            // The first codebook head, as [vocabulary, width], stands in for the vocabulary of the depth decoder.
            collector.Add(HeadStandIn(heads, width, vocabulary, "depth.token_embedding.weight"));
            collector.Add(HeadStandIn(heads, width, vocabulary, "depth.lm_head.weight"));
        }

        var codecCollector = new HfWeightCollector();
        MimiHfMapping.MapInto(codecCollector, codec);
        foreach (var (name, w) in codecCollector.Weights)
        {
            collector.Add(w.As("codec." + name));
        }

        collector.ThrowIfUnmapped(c.ModelType);
        return collector.Weights;
    }

    // Head 0 of the codebook heads ([31, width, vocabulary]) as [vocabulary, width] f32.
    private static WeightInfo HeadStandIn(WeightInfo heads, long width, long vocabulary, string name)
        => WeightInfo.Derived(name, DataType.F32, new Shape(vocabulary, width), () =>
        {
            int elementBytes = heads.Type == DataType.F32 ? 4 : 2;
            var raw = new byte[width * vocabulary * elementBytes];
            heads.Source.Read(0, raw);
            var values = new float[width * vocabulary];
            for (long k = 0; k < width; k++)
            {
                for (long o = 0; o < vocabulary; o++)
                {
                    long i = (k * vocabulary) + o;
                    values[(o * width) + k] = heads.Type == DataType.F32
                        ? BitConverter.ToSingle(raw, (int)(i * 4))
                        : heads.Type == DataType.BF16
                            ? BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadUInt16LittleEndian(raw.AsSpan((int)(i * 2))) << 16)
                            : (float)BinaryPrimitives.ReadHalfLittleEndian(raw.AsSpan((int)(i * 2)));
                }
            }

            return HfTransforms.EncodeF32(values);
        });

    private static HfConfig BackboneConfig(HfConfig c)
    {
        var node = JsonNode.Parse(c.Root.GetRawText())!.AsObject();
        node["model_type"] = "llama";
        node["architectures"] = new JsonArray("LlamaForCausalLM");
        node["tie_word_embeddings"] = false;
        node.Remove("codec_config");
        node.Remove("depth_decoder_config");
        return HfConfig.Parse(node.ToJsonString(), c.Directory);
    }

    private static HfConfig DepthConfig(HfConfig c)
    {
        var node = JsonNode.Parse(c.Root.GetProperty("depth_decoder_config").GetRawText())!.AsObject();
        node["model_type"] = "llama";
        node["architectures"] = new JsonArray("LlamaForCausalLM");
        node["tie_word_embeddings"] = false;
        return HfConfig.Parse(node.ToJsonString(), c.Directory);
    }

    private static HfConfig CodecConfig(HfConfig c)
        => HfConfig.Parse(c.Root.GetProperty("codec_config").GetRawText(), c.Directory);
}
