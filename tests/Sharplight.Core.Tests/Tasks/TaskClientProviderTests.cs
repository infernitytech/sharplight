using Microsoft.Extensions.DependencyInjection;
using Sharplight.Audio;
using Sharplight.ImageGeneration;
using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Speech;
using Sharplight.Tasks;
using Xunit;

namespace Sharplight.Core.Tests.Tasks;

public class TaskClientProviderTests
{
    [Fact]
    public void ResolvesFactoriesByKindAndClientTypeAndTheLastRegistrationWins()
    {
        using var services = new ServiceCollection()
            .AddSharplightTasks()
            .AddTaskClientFactory<ISpeechRecognizer>(ModelKinds.SpeechRecognition, m => new FakeRecognizer(m, "generic"))
            .AddTaskClientFactory<ISpeechRecognizer>(ModelKinds.SpeechRecognition, m => new FakeRecognizer(m, "special"), m => m.Definition.Architecture == "special")
            .BuildServiceProvider();
        var provider = services.GetRequiredService<TaskClientProvider>();
        using var generic = Model(ModelKinds.SpeechRecognition, "plain");
        using var special = Model(ModelKinds.SpeechRecognition, "special");

        using (var client = provider.Create<ISpeechRecognizer>(generic))
        {
            Assert.Equal("generic", ((FakeRecognizer)client).Variant);
            Assert.Equal(new TaskClientMetadata(ModelKinds.SpeechRecognition, "test", "plain"), client.Metadata);
            Assert.Same(generic, client.GetService(typeof(TaskModel)));
        }

        using (var client = provider.Create<ISpeechRecognizer>(special))
        {
            Assert.Equal("special", ((FakeRecognizer)client).Variant);
        }

        Assert.Contains(ModelKinds.SpeechRecognition, provider.Kinds);
        Assert.Contains(ModelKinds.TextGeneration, provider.Kinds);
    }

    [Fact]
    public void UnservedKindsAndClientTypesAreReported()
    {
        using var services = new ServiceCollection()
            .AddSharplightTasks()
            .AddTaskClientFactory<ISpeechRecognizer>(ModelKinds.SpeechRecognition, m => new FakeRecognizer(m, "generic"))
            .BuildServiceProvider();
        var provider = services.GetRequiredService<TaskClientProvider>();
        using var depth = Model("depth-estimation", "dpt");
        using var speech = Model(ModelKinds.SpeechRecognition, "plain");

        Assert.False(provider.CanCreate<ISpeechRecognizer>(depth));
        Assert.Contains("depth-estimation", Assert.Throws<NotSupportedException>(() => provider.Create<ISpeechRecognizer>(depth)).Message);
        Assert.False(provider.CanCreate<IImageGenerator>(speech));
        Assert.Throws<NotSupportedException>(() => provider.Create<IImageGenerator>(speech));
        Assert.True(provider.CanCreate<ITaskClient>(speech));
    }

    [Fact]
    public void BuiltInTextGenerationServesDecoderModelsOnly()
    {
        using var services = new ServiceCollection().AddSharplightTasks().AddSharplightTasks().BuildServiceProvider();
        var provider = services.GetRequiredService<TaskClientProvider>();
        // Registering twice adds each built-in factory once (text generation, speech recognition, the image encoders).
        var factories = services.GetServices<ITaskClientFactory>().ToList();
        using var once = new ServiceCollection().AddSharplightTasks().BuildServiceProvider();
        Assert.Equal(once.GetServices<ITaskClientFactory>().Count(), factories.Count);
        Assert.Single(factories, f => f.Kind == ModelKinds.TextGeneration);
        Assert.Single(factories, f => f.Kind == ModelKinds.SpeechRecognition);
        using var decoder = Model(null, "llama", WellKnown.DecodeGraph);
        using var graphOnly = Model(null, "onnx", "main");
        Assert.True(provider.CanCreate<ITextGenerator>(decoder));
        Assert.False(provider.CanCreate<ITextGenerator>(graphOnly));
    }

    [Fact]
    public void ClientsRegisteredPerModelUseNamedLoadOptions()
    {
        var services = new ServiceCollection()
            .AddSpeechRecognizer(o => o.ModelPath = "a.slm", serviceKey: "a")
            .AddSpeechRecognizer(o => o.ModelPath = "b.slm", serviceKey: "b")
            .AddTextGenerator(o => o.ContextLength = 4096);
        using var provider = services.BuildServiceProvider();
        var monitor = provider.GetRequiredService<Microsoft.Extensions.Options.IOptionsMonitor<TaskLoadOptions>>();
        Assert.Equal("a.slm", monitor.Get(typeof(ISpeechRecognizer).FullName + ":a").ModelPath);
        Assert.Equal("b.slm", monitor.Get(typeof(ISpeechRecognizer).FullName + ":b").ModelPath);
        Assert.Equal(4096, monitor.Get(typeof(ITextGenerator).FullName).ContextLength);
        Assert.Single(provider.GetServices<TaskClientProvider>());
        // Resolving would load the model; without devices loading is refused with a clear message.
        var error = Assert.Throws<ArgumentException>(() => provider.GetRequiredKeyedService<ISpeechRecognizer>("a"));
        Assert.Contains("Devices", error.Message);
    }

    internal static TaskModel Model(string? kind, string architecture, params string[] graphs)
    {
        var graphMap = (graphs.Length == 0 ? ["encode"] : graphs).ToDictionary(g => g, TestGraphs.Trivial);
        var definition = new ModelDefinition
        {
            Architecture = architecture,
            Name = "test",
            Weights = new Dictionary<string, WeightInfo>(),
            Graphs = graphMap,
            Description = kind is null ? null : new ModelDescription { Kind = kind, Components = [new ModelComponent { Role = "model", Graphs = [.. graphMap.Keys] }] },
        };
        var source = new ModelSource
        {
            Format = "test",
            Architecture = architecture,
            Metadata = new Dictionary<string, object>(),
            Weights = new Dictionary<string, WeightInfo>(),
            Definition = definition,
        };
        return new TaskModel(definition, source, new TaskLoadOptions(), new ArchitectureCatalog([]));
    }

    private sealed class FakeRecognizer(TaskModel model, string variant) : ISpeechRecognizer
    {
        public string Variant => variant;

        public TaskClientMetadata Metadata { get; } = TaskClientMetadata.For(model.Definition);

        public int SampleRate => 16000;

        public object? GetService(Type serviceType, object? serviceKey = null) => serviceType.IsInstanceOfType(model) ? model : null;

        public Task<Transcription> TranscribeAsync(AudioBuffer audio, SpeechRecognitionOptions? options = null, CancellationToken cancellationToken = default)
            => Task.FromResult(new Transcription("", null, [], audio.Duration));

        public async IAsyncEnumerable<TranscriptionSegment> TranscribeStreamingAsync(AudioBuffer audio, SpeechRecognitionOptions? options = null,
            [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            await Task.CompletedTask;
            yield break;
        }

        public void Dispose()
        {
        }
    }
}
