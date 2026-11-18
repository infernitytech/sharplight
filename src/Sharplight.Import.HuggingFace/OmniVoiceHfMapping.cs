using System.Text.Json.Nodes;
using Sharplight.Models;
using Sharplight.Vision;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// OmniVoice (<c>omnivoice</c>, k2-fsa: a Qwen3 backbone over text tokens and eight audio codebooks, decoded by masked diffusion) → canonical <c>omnivoice</c> (documented on
/// <c>OmniVoiceAdapter</c>). The backbone (<c>llm.</c>) maps through <see cref="DecoderHfMapping"/> as a bidirectional model whose rows are embedded by the host (<c>decoder.embedding_input</c>),
/// with the eight audio heads as the output projection (<c>lm_head</c>, 8 × 1025 rows) and the text embedding kept as the token embedding; the audio embedding table keeps its name under
/// <c>omnivoice.</c>. The audio tokenizer in <c>audio_tokenizer/</c> is the model's vocoder companion (<see cref="HiggsAudioTokenizerHfMapping"/>).
/// </summary>
internal sealed class OmniVoiceHfMapping : IHfArchitectureMapping
{
    /// <summary>The folder of the audio tokenizer inside the model folder.</summary>
    public const string AudioTokenizerFolder = "audio_tokenizer";

    private const string Llm = "llm.";

    /// <inheritdoc/>
    public string Architecture => "omnivoice";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["omnivoice"];

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HuggingFaceImportOptions options)
    {
        var md = new Dictionary<string, object>(StringComparer.Ordinal);
        using (var config = LlmConfig(c))
        {
            foreach (var (key, value) in new DecoderHfMapping().MapHyperParameters(config, options))
            {
                md[key] = value;
            }
        }

        DreamHfMapping.AddDiffusion(md, c, shiftLogits: false);
        string a = Architecture + ".";
        int codebooks = c.Int("num_audio_codebook");
        md[a + "codebooks"] = codebooks;
        md[a + "audio_vocab"] = c.Int("audio_vocab_size");
        md[a + "mask_id"] = c.Int("audio_mask_id");
        md[a + "codebook_weights"] = c.Root.GetProperty("audio_codebook_weights").EnumerateArray().Select(v => (float)v.GetDouble()).ToArray();
        return md;
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
    {
        var collector = new HfWeightCollector();
        var llm = new Dictionary<string, SafetensorsTensor>(StringComparer.Ordinal);
        foreach (var (name, tensor) in tensors)
        {
            if (name.StartsWith(Llm, StringComparison.Ordinal))
            {
                llm["model." + name[Llm.Length..]] = tensor;
            }
            else if (name == "audio_heads.weight")
            {
                llm["lm_head.weight"] = tensor;
            }
            else if (name == "audio_embeddings.weight")
            {
                collector.Add(HfTransforms.Plain(tensor.ToWeightInfo(), Architecture + ".audio_embedding.weight"));
            }
            else if (name != "codebook_layer_offsets")
            {
                collector.AddUnmapped(name);
            }
        }

        using (var config = LlmConfig(c))
        {
            foreach (var (_, w) in new DecoderHfMapping().MapWeights(config, llm, options))
            {
                collector.Add(w);
            }
        }

        collector.ThrowIfUnmapped(c.ModelType);
        return collector.Weights;
    }

    /// <inheritdoc/>
    public void AddMetadata(IDictionary<string, object> metadata, HfConfig config, string directory) => metadata["decoder.embedding_input"] = true;

    /// <summary>The audio tokenizer folder as the vocoder companion.</summary>
    /// <param name="directory">The model folder.</param>
    public IReadOnlyDictionary<string, string> FindCompanions(string directory)
    {
        string codec = Path.Combine(directory, AudioTokenizerFolder);
        return File.Exists(Path.Combine(codec, "config.json"))
            ? new Dictionary<string, string> { [AuxiliaryRoles.Vocoder] = codec }
            : new Dictionary<string, string>();
    }

    // A Qwen3 causal-LM configuration of the backbone (the head is the audio heads').
    private static HfConfig LlmConfig(HfConfig c)
    {
        var node = JsonNode.Parse(c.Root.GetProperty("llm_config").GetRawText())!.AsObject();
        node["model_type"] = "qwen3";
        node["architectures"] = new JsonArray("Qwen3ForCausalLM");
        node["tie_word_embeddings"] = false;
        return HfConfig.Parse(node.ToJsonString(), c.Directory);
    }
}
