using System.Text.Json;
using Sharplight.Graph;
using Sharplight.Models;
using Xunit;

namespace Sharplight.Core.Tests.Models;

public class ModelDescriptionTests
{
    internal static JsonElement Json(string text) => JsonDocument.Parse(text).RootElement.Clone();

    internal static ModelDescription SpeechRecognizer() => new()
    {
        Kind = ModelKinds.SpeechRecognition,
        Config = new Dictionary<string, JsonElement> { ["sampleRate"] = Json("16000") },
        Components =
        [
            new ModelComponent
            {
                Role = ModelComponentRoles.Encoder,
                Graphs = ["encode"],
                WeightPrefixes = ["encoder."],
                Config = new Dictionary<string, JsonElement> { ["layers"] = Json("4"), ["activation"] = Json("\"gelu\"") },
                SymbolBounds = new Dictionary<string, long> { ["F"] = 3000 },
            },
            new ModelComponent { Role = ModelComponentRoles.Decoder, Graphs = ["decode"], WeightPrefixes = ["decoder."] },
            new ModelComponent
            {
                Role = ModelComponentRoles.Tokenizer,
                MetadataPrefix = "tokenizer.",
                Attachments = new Dictionary<string, string> { ["vocab"] = "tokenizer/vocab.json" },
            },
            new ModelComponent
            {
                Role = ModelComponentRoles.FeatureExtractor,
                Config = new Dictionary<string, JsonElement> { ["melBins"] = Json("80"), ["hop"] = Json("{\"length\": 160, \"window\": [1.5, 2]}") },
                Attachments = new Dictionary<string, string> { ["mel-filters"] = "feature-extractor/mel_filters.f32" },
            },
        ],
    };

    [Fact]
    public void JsonRoundTripKeepsEverything()
    {
        var description = SpeechRecognizer();
        var json = description.ToJson();
        var read = ModelDescription.FromJson(json);

        Assert.Equal(json, read.ToJson());
        Assert.Equal(ModelDescription.CurrentSchemaVersion, read.SchemaVersion);
        Assert.Equal(ModelKinds.SpeechRecognition, read.Kind);
        Assert.Equal(["encoder", "decoder", "tokenizer", "feature-extractor"], read.Components.Select(c => c.Role));
        var encoder = read.Component(ModelComponentRoles.Encoder);
        Assert.Equal(["encode"], encoder.Graphs);
        Assert.Equal(3000, encoder.SymbolBounds["F"]);
        Assert.Equal(4, encoder.ConfigValue("layers", 0));
        Assert.Equal("gelu", encoder.ConfigValue("activation", ""));
        Assert.True(encoder.OwnsWeight("encoder.blocks.0.w"));
        Assert.False(encoder.OwnsWeight("decoder.blocks.0.w"));
        Assert.Equal(16000, read.ConfigValue("sampleRate", 0));
        var hop = read.Component(ModelComponentRoles.FeatureExtractor).Config["hop"];
        Assert.Equal(160, hop.GetProperty("length").GetInt32());
        Assert.Equal("tokenizer/vocab.json", read.Component(ModelComponentRoles.Tokenizer).Attachments["vocab"]);
        Assert.Contains("\"schemaVersion\": 1", json);
    }

    [Fact]
    public void ReaderIgnoresUnknownFieldsAndRejectsNewerSchemas()
    {
        var withExtras = ModelDescription.FromJson("""{ "schemaVersion": 1, "kind": "time-series", "futureField": [1, 2], "components": [{ "role": "model", "newThing": true }] }""");
        Assert.Equal("time-series", withExtras.Kind);
        Assert.Equal("model", Assert.Single(withExtras.Components).Role);
        Assert.Throws<NotSupportedException>(() => ModelDescription.FromJson("""{ "schemaVersion": 2, "kind": "x" }"""));
        Assert.Throws<InvalidDataException>(() => ModelDescription.FromJson("not json"));
        Assert.Throws<InvalidDataException>(() => ModelDescription.FromJson("""{ "kind": "" }"""));
        Assert.Throws<InvalidDataException>(() => ModelDescription.FromJson("""{ "kind": "x", "components": [{ "role": "a" }, { "role": "a" }] }"""));
    }

    [Fact]
    public void ValidateChecksGraphsAndAttachmentsOfTheDefinition()
    {
        var definition = Definition(["encode", "decode"], ["tokenizer/vocab.json", "feature-extractor/mel_filters.f32"]);
        SpeechRecognizer().Validate(definition);
        var missingGraph = Definition(["encode"], ["tokenizer/vocab.json", "feature-extractor/mel_filters.f32"]);
        Assert.Contains("'decode'", Assert.Throws<InvalidDataException>(() => SpeechRecognizer().Validate(missingGraph)).Message);
        var missingAttachment = Definition(["encode", "decode"], ["tokenizer/vocab.json"]);
        Assert.Contains("mel_filters", Assert.Throws<InvalidDataException>(() => SpeechRecognizer().Validate(missingAttachment)).Message);
    }

    [Fact]
    public void ModelsWithoutDescriptionResolveToSingleComponentTextGeneration()
    {
        var definition = Definition(["decode", "decode_multimodal"], []);
        var resolved = ModelDescription.Resolve(definition);
        Assert.Equal(ModelKinds.TextGeneration, resolved.Kind);
        var component = Assert.Single(resolved.Components);
        Assert.Equal(ModelComponentRoles.Decoder, component.Role);
        Assert.Equal(["decode", "decode_multimodal"], component.Graphs);
        Assert.False(resolved.TryGetComponent("encoder", out _));
        Assert.Throws<KeyNotFoundException>(() => resolved.Component("encoder"));

        var described = new ModelDefinition
        {
            Architecture = "x",
            Weights = new Dictionary<string, WeightInfo>(),
            Graphs = new Dictionary<string, ComputeGraph>(),
            Description = SpeechRecognizer(),
        };
        Assert.Same(described.Description, ModelDescription.Resolve(described));
    }

    [Fact]
    public void ComponentMetadataIsScopedByPrefix()
    {
        var metadata = new Dictionary<string, object> { ["tokenizer.tokens"] = new[] { "a" }, ["general.name"] = "m" };
        var tokenizer = SpeechRecognizer().Component(ModelComponentRoles.Tokenizer);
        Assert.Equal(["tokens"], tokenizer.Metadata(metadata).Keys);
        Assert.Same(metadata, SpeechRecognizer().Component(ModelComponentRoles.Encoder).Metadata(metadata));
    }

    private static ModelDefinition Definition(string[] graphs, string[] attachments) => new()
    {
        Architecture = "test",
        Weights = new Dictionary<string, WeightInfo>(),
        Graphs = graphs.ToDictionary(g => g, TestGraphs.Trivial),
        Attachments = attachments.ToDictionary(a => a, _ => (IWeightSource)new ByteSource([1, 2, 3])),
    };

    private sealed class ByteSource(byte[] bytes) : IWeightSource
    {
        public long ByteLength => bytes.Length;

        public void Read(long offset, Span<byte> destination) => bytes.AsSpan((int)offset, destination.Length).CopyTo(destination);
    }
}
