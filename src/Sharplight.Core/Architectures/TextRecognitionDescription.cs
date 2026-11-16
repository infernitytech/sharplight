using System.Text.Json;
using Sharplight.Models;
using Sharplight.Tokenizers;

namespace Sharplight.Architectures;

/// <summary>
/// Builds the <see cref="ModelDescription"/> of an image-to-text recognizer on the encoder-decoder (kind <see cref="ModelKinds.Ocr"/>): the image
/// <c>encoder</c> (patchified image in, encoder rows out), the <c>decoder</c> (cross-attention over the cached encoder rows, KV-cached self-attention),
/// the <c>tokenizer</c> component (the tokenizer is stored as an attachment, the canonical <see cref="TokenizerData"/> JSON) and the
/// <c>feature-extractor</c> with the model's <see cref="Imaging.ImageProcessorConfig"/>. The decoder component's configuration carries
/// <c>startToken</c> and <c>endToken</c> (canonical keys <c>{arch}.decoder_start_token</c> and <c>{arch}.end_token</c>).
/// </summary>
internal static class TextRecognitionDescription
{
    /// <summary>The attachment that stores the tokenizer.</summary>
    public const string TokenizerAttachment = "tokenizer/tokenizer.json";

    /// <summary>The description, the definition's attachments and its metadata without the tokenizer keys.</summary>
    /// <param name="src">The model source (hyper-parameters, tokenizer metadata, image processor).</param>
    /// <param name="decoderContext">The decoder's context length in tokens.</param>
    /// <param name="sourceRows">Encoder output rows.</param>
    public static (ModelDescription Description, IReadOnlyDictionary<string, object> Metadata, IReadOnlyDictionary<string, IWeightSource> Attachments) Build(ModelSource src, int decoderContext, int sourceRows)
        => Make(src, decoderContext, sourceRows);

    private static (ModelDescription, IReadOnlyDictionary<string, object>, IReadOnlyDictionary<string, IWeightSource>) Make(ModelSource src, int decoderContext, int sourceRows)
    {
        var tokenizer = TokenizerData.FromMetadata(src.Metadata);
        var attachments = new Dictionary<string, IWeightSource>(src.Attachments, StringComparer.Ordinal)
        {
            [TokenizerAttachment] = new MemoryWeightSource(tokenizer.ToJson()),
        };
        var metadata = src.Metadata.Where(e => !e.Key.StartsWith("tokenizer.", StringComparison.Ordinal)).ToDictionary(StringComparer.Ordinal);
        var processor = System.Text.Json.Nodes.JsonNode.Parse(src.Get<string>(CanonicalKeys.ImageProcessor))!;
        bool promptEncoder = src.Arch("encoder.input", "image") == "image_text";
        var config = new Dictionary<string, JsonElement>(StringComparer.Ordinal);
        if (promptEncoder)
        {
            config["tasks"] = JsonSerializer.SerializeToElement(System.Text.Json.Nodes.JsonNode.Parse(src.Arch<string>("prompt.tasks")));
            config["promptBos"] = JsonSerializer.SerializeToElement(src.Arch("prompt.bos_token", 0));
            config["promptEos"] = JsonSerializer.SerializeToElement(src.Arch("prompt.eos_token", 2));
            config["quadTasks"] = JsonSerializer.SerializeToElement(src.Arch("prompt.quad_tasks", "").Split(',', StringSplitOptions.RemoveEmptyEntries));
            config["locationBins"] = JsonSerializer.SerializeToElement(src.Arch("prompt.location_bins", 1000));
        }

        var description = new ModelDescription
        {
            Kind = ModelKinds.Ocr,
            Config = config,
            Components =
            [
                new ModelComponent
                {
                    Role = ModelComponentRoles.Encoder,
                    Graphs = [WellKnown.EncodeGraph],
                    WeightPrefixes = promptEncoder ? ["vision.", "encoder.", "token_embedding."] : ["vision."],
                    SymbolBounds = promptEncoder ? new Dictionary<string, long>(StringComparer.Ordinal) { [WellKnown.TokenSymbol] = src.Arch("prompt.max_tokens", 128) } : new Dictionary<string, long>(StringComparer.Ordinal),
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.Decoder,
                    Graphs = [WellKnown.DecodeGraph],
                    WeightPrefixes = ["decoder.", "token_embedding.", "output."],
                    SymbolBounds = new Dictionary<string, long>(StringComparer.Ordinal)
                    {
                        [WellKnown.TokenSymbol] = decoderContext,
                        [WellKnown.ContextSymbol] = decoderContext,
                        [WellKnown.EncoderSymbol] = sourceRows,
                        ["R"] = 1,
                    },
                    Config = ComponentConfig.Of(("startToken", src.Arch("decoder_start_token", 0)), ("endToken", src.Arch("end_token", 2)), ("contextLength", decoderContext),
                        ("forcedFirstToken", src.Arch("forced_bos_token", -1)), ("noRepeatNgram", src.Arch("no_repeat_ngram_size", 0))),
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.Tokenizer,
                    Attachments = new Dictionary<string, string>(StringComparer.Ordinal) { [ModelComponentAttachments.Tokenizer] = TokenizerAttachment },
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.FeatureExtractor,
                    Config = new Dictionary<string, JsonElement> { [Imaging.ImageProcessor.ComponentKey] = JsonSerializer.SerializeToElement(processor) },
                },
            ],
        };
        return (description, metadata, attachments);
    }
}
