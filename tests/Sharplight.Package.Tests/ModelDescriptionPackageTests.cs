using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Sharplight.Audio;
using Sharplight.Cpu;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Speech;
using Sharplight.Tasks;
using Sharplight.Tensors;
using Xunit;

namespace Sharplight.Package.Tests;

/// <summary>The model description (DESC) and attachments (ATCH) of multi-component packages, and how the task layer loads them.</summary>
public class ModelDescriptionPackageTests
{
    private const string Vocab = "{\"hello\": 0}";
    private static readonly Lazy<IComputeDevice> s_cpu = new(() =>
    {
        var provider = new CpuExecutionProvider();
        return provider.OpenDevice(provider.GetDevices()[0]);
    });

    [Fact]
    public void DescriptionAndAttachmentsRoundTrip()
    {
        using var tmp = new TempDir();
        var definition = SpeechDefinition();
        var path = tmp.File("speech.slm");
        ModelPackage.Save(path, definition, []);

        using (var file = new PackageFile(path))
        {
            var stored = file.Definition;
            Assert.NotNull(stored.Description);
            Assert.Equal(definition.Description!.ToJson(), stored.Description.ToJson());
            Assert.Equal(["tokenizer/vocab.json", "feature-extractor/mel_filters.f32"], stored.Attachments.Keys.Order(StringComparer.Ordinal).Reverse());
            var vocab = stored.Attachments["tokenizer/vocab.json"];
            var bytes = new byte[vocab.ByteLength];
            vocab.Read(0, bytes);
            Assert.Equal(Vocab, Encoding.UTF8.GetString(bytes));
            Assert.Equal(0, vocab.TryGetPointer() % 4096);
            var filters = stored.Attachments["feature-extractor/mel_filters.f32"];
            Assert.Equal(MelFilters().Length, filters.ByteLength);
        }

        var info = ModelPackageReader.Open(path);
        Assert.True(info.Definition.HasDescription);
        Assert.Equal(ModelKinds.SpeechRecognition, info.Definition.Description.Kind);
        Assert.Equal(["encoder", "decoder", "tokenizer", "feature-extractor"], info.Definition.Description.Components.Select(c => c.Role));
        Assert.Equal(2, info.Attachments.Count);
        Assert.All(info.Attachments, a => Assert.Equal(0, a.Offset % (64 * 1024)));
        Assert.Contains(info.Sections, s => s.Kind == "DESC" && s.Name == "model.json");
        Assert.Equal(2, info.Sections.Count(s => s.Kind == "ATCH"));
        Assert.Contains("DESC", ModelPackageReader.SectionKinds);
        Assert.Contains("ATCH", ModelPackageReader.SectionKinds);
    }

    [Fact]
    public void PackagesWithoutDescriptionStillLoadAsTextGeneration()
    {
        using var tmp = new TempDir();
        var path = tmp.File("llm.slm");
        ModelPackage.Save(path, PackageTests.SmallDefinition(), []);

        var info = ModelPackageReader.Open(path);
        Assert.Equal(["DEFN", "WIDX", "WDAT"], info.Sections.Select(s => s.Kind));
        Assert.False(info.Definition.HasDescription);
        Assert.Equal(ModelKinds.TextGeneration, info.Definition.Description.Kind);
        Assert.Empty(info.Attachments);

        var source = new ModelPackageImporter().Open(path);
        try
        {
            var definition = ModelAdapterRegistry.Build(new ModelSourceSet { Primary = source });
            Assert.Null(definition.Description);
            Assert.Empty(definition.Attachments);
            var resolved = ModelDescription.Resolve(definition);
            Assert.Equal(ModelKinds.TextGeneration, resolved.Kind);
            Assert.Equal(["decode"], Assert.Single(resolved.Components).Graphs);
        }
        finally
        {
            foreach (var resource in source.OwnedResources)
            {
                resource.Dispose();
            }
        }
    }

    [Fact]
    public void CompactAndArtifactsKeepDescriptionAndAttachments()
    {
        using var tmp = new TempDir();
        var path = tmp.File("speech.slm");
        ModelPackage.Save(path, SpeechDefinition(), []);
        var set = new CompilationCache(FakeArtifactDevice.Code());
        set.Put("kernel/x", new byte[100]);
        ModelPackage.AddArtifacts(path, [set]);
        ModelPackage.AddArtifacts(path, [set]);
        Assert.True(ModelPackage.Compact(path) > 0);

        using var file = new PackageFile(path);
        Assert.Equal(ModelKinds.SpeechRecognition, file.Definition.Description!.Kind);
        Assert.Equal(2, file.Definition.Attachments.Count);
        Assert.Single(file.ArtifactSets);
    }

    [Fact]
    public void InconsistentDescriptionsAreRejectedOnSave()
    {
        using var tmp = new TempDir();
        var definition = SpeechDefinition();
        var broken = new ModelDefinition
        {
            Architecture = definition.Architecture,
            Weights = definition.Weights,
            Graphs = definition.Graphs,
            Attachments = new Dictionary<string, IWeightSource>(),
            Description = definition.Description,
        };
        var error = Assert.Throws<InvalidDataException>(() => ModelPackage.Save(tmp.File("broken.slm"), broken, []));
        Assert.Contains("attachment", error.Message);
        Assert.False(File.Exists(tmp.File("broken.slm")));
    }

    [Fact]
    public async Task TaskClientsResolvedFromTheContainerRunTheComponentsOfAPackage()
    {
        using var tmp = new TempDir();
        var path = tmp.File("speech.slm");
        ModelPackage.Save(path, SpeechDefinition(), []);
        ModelPackageImporter.Register();

        EncoderOnlyRecognizer? created = null;
        var services = new ServiceCollection()
            .AddSharplightTasks()
            .AddTaskClientFactory<ISpeechRecognizer>(ModelKinds.SpeechRecognition, model => created = new EncoderOnlyRecognizer(model))
            .AddSpeechRecognizer(options =>
            {
                options.ModelPath = path;
                options.Devices = [new DeviceSlot { Device = s_cpu.Value }];
            });
        var audio = new AudioBuffer([.. Enumerable.Range(0, 32).Select(i => i / 32f)], 16000);
        await using (var provider = services.BuildServiceProvider())
        {
            var recognizer = provider.GetRequiredService<ISpeechRecognizer>();
            Assert.Same(recognizer, provider.GetRequiredService<ISpeechRecognizer>());
            Assert.Equal(new TaskClientMetadata(ModelKinds.SpeechRecognition, "tiny-speech", "test_speech"), recognizer.Metadata);
            Assert.Equal(16000, recognizer.SampleRate);
            var model = (TaskModel)recognizer.GetService(typeof(TaskModel))!;
            Assert.Equal(["encoder", "decoder", "tokenizer", "feature-extractor"], model.Description.Components.Select(c => c.Role));

            var transcription = await recognizer.TranscribeAsync(audio, cancellationToken: TestContext.Current.CancellationToken);
            Assert.Equal(FormattableString.Invariant($"{Vocab}:{ExpectedEncoderSum(audio):F3}"), transcription.Text);
        }

        Assert.True(created!.Disposed);
    }

    [Fact]
    public async Task PrecompilationCompilesEveryComponentGraph()
    {
        using var tmp = new TempDir();
        var path = tmp.File("speech.slm");
        ModelPackage.Save(path, SpeechDefinition(), []);
        var provider = new FakeCodeProvider(s_cpu.Value, ["1"]);
        var result = await PackagePrecompiler.PrecompileAsync(path, [provider], new PrecompileOptions
        {
            SymbolBounds = new Dictionary<string, long> { ["T"] = 4 },
            Tiers = new HashSet<string> { ArtifactTiers.Code },
        }, cancellationToken: TestContext.Current.CancellationToken);

        Assert.Single(result.StoredSets);
        using var file = new PackageFile(path);
        var set = Assert.Single(file.ArtifactSets);
        Assert.True(set.Entries.TryGet("kernel/encode/0", out _));
        Assert.True(set.Entries.TryGet("kernel/decode/0", out _));
    }

    private static double ExpectedEncoderSum(AudioBuffer audio)
    {
        var weights = EncoderWeights();
        double sum = 0;
        for (int frame = 0; frame < EncoderOnlyRecognizer.Frames; frame++)
        {
            for (int row = 0; row < 4; row++)
            {
                for (int k = 0; k < EncoderOnlyRecognizer.FeatureSize; k++)
                {
                    sum += audio.Samples[frame * EncoderOnlyRecognizer.FeatureSize + k] * weights[row * EncoderOnlyRecognizer.FeatureSize + k];
                }
            }
        }

        return sum;
    }

    private static float[] EncoderWeights() => [.. Enumerable.Range(0, 4 * 8).Select(i => 0.1f * (i / 8 + 1))];

    private static byte[] MelFilters() => MemoryMarshal.AsBytes(Enumerable.Range(0, 80 * 3).Select(i => (float)i).ToArray().AsSpan()).ToArray();

    private static WeightInfo Weight(string name, Shape shape, float[] values)
        => WeightInfo.Derived(name, DataType.F32, shape, () => MemoryMarshal.AsBytes(values.AsSpan()).ToArray());

    /// <summary>A two-graph speech-recognition-shaped model: encoder and decoder graphs, a tokenizer and a feature extractor.</summary>
    internal static ModelDefinition SpeechDefinition()
    {
        var weights = new Dictionary<string, WeightInfo>
        {
            ["encoder.w"] = Weight("encoder.w", new Shape(4, 8), EncoderWeights()),
            ["decoder.w"] = Weight("decoder.w", new Shape(2, 4), [.. Enumerable.Repeat(0.5f, 8)]),
        };
        var encode = new GraphBuilder("encode", n => weights.GetValueOrDefault(n));
        var features = encode.Input("features", InputRole.Other, DataType.F32, new Shape(Dim.Sym("F"), 8));
        encode.Output("hidden", encode.MatMul(features, encode.Weight("encoder.w"), "hidden"));
        var decode = new GraphBuilder("decode", n => weights.GetValueOrDefault(n));
        var hidden = decode.Input("hidden", InputRole.Embeddings, DataType.F32, new Shape(Dim.Sym("T"), 4));
        decode.Output("logits", decode.MatMul(hidden, decode.Weight("decoder.w"), "logits"));

        static JsonElement Json(string text) => JsonDocument.Parse(text).RootElement.Clone();
        var description = new ModelDescription
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
                    SymbolBounds = new Dictionary<string, long> { ["F"] = 64 },
                },
                new ModelComponent { Role = ModelComponentRoles.Decoder, Graphs = ["decode"], WeightPrefixes = ["decoder."] },
                new ModelComponent
                {
                    Role = ModelComponentRoles.Tokenizer,
                    Attachments = new Dictionary<string, string> { ["vocab"] = "tokenizer/vocab.json" },
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.FeatureExtractor,
                    Config = new Dictionary<string, JsonElement> { ["melBins"] = Json("80"), ["hopLength"] = Json("160") },
                    Attachments = new Dictionary<string, string> { ["mel-filters"] = "feature-extractor/mel_filters.f32" },
                },
            ],
        };
        return new ModelDefinition
        {
            Architecture = "test_speech",
            Name = "tiny-speech",
            Weights = weights,
            Graphs = new Dictionary<string, ComputeGraph> { ["encode"] = encode.Build(), ["decode"] = decode.Build() },
            Description = description,
            Attachments = new Dictionary<string, IWeightSource>
            {
                ["tokenizer/vocab.json"] = new BytesSource(Encoding.UTF8.GetBytes(Vocab)),
                ["feature-extractor/mel_filters.f32"] = new BytesSource(MelFilters()),
            },
        };
    }
}
