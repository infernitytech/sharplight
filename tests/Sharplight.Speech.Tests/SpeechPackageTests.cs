using Microsoft.Extensions.DependencyInjection;
using Sharplight.Audio;
using Sharplight.Models;
using Sharplight.Package;
using Sharplight.Runtime;
using Sharplight.Tasks;
using Sharplight.Vulkan;
using Xunit;

namespace Sharplight.Speech.Tests;

/// <summary>
/// Speech-recognition models in <c>.slm</c> packages: conversion stores the description (kind and components) and the attachments,
/// the package loads through the task layer and transcribes exactly like its source, and precompilation covers both component graphs.
/// </summary>
public sealed class SpeechPackageTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "sharplight-speech-" + Guid.NewGuid().ToString("N"));

    /// <summary>Creates the scratch directory.</summary>
    public SpeechPackageTests() => Directory.CreateDirectory(_directory);

    /// <summary>A converted package describes the model and transcribes like the source.</summary>
    [Theory]
    [InlineData("whisper.cpp", "ggml-tiny.en-q5_1", "cpu")]
    [InlineData("whisper.cpp", "ggml-tiny", "vulkan")]
    [InlineData("hf", "whisper-tiny.en", "cpu")]
    public async Task PackageTranscribesLikeItsSource(string format, string name, string device)
    {
        string source = format == "hf" ? SpeechEnvironment.HuggingFace(name) : SpeechEnvironment.WhisperCpp(name);
        if (!SpeechEnvironment.Available(source, device))
        {
            Assert.Skip($"{source}, jfk.wav or the {device} device is missing.");
        }

        string package = Path.Combine(_directory, name + ".slm");
        ModelConverter.Convert(source, package);

        var info = ModelPackageReader.Open(package);
        Assert.True(info.Definition.HasDescription);
        var description = info.Definition.Description;
        Assert.Equal(ModelKinds.SpeechRecognition, description.Kind);
        Assert.Equal(
            [ModelComponentRoles.Encoder, ModelComponentRoles.Decoder, ModelComponentRoles.Tokenizer, ModelComponentRoles.FeatureExtractor],
            description.Components.Select(c => c.Role));
        Assert.Contains(info.Attachments, a => a.Name == "feature-extractor/mel-filters");
        Assert.Contains(info.Attachments, a => a.Name == "tokenizer/tokenizer.json");

        var audio = WavFile.Read(SpeechEnvironment.Jfk);
        var options = new SpeechRecognitionOptions { Language = name.Contains(".en", StringComparison.Ordinal) ? null : "en" };
        var expected = await Transcribe(source, device, audio, options);
        var actual = await Transcribe(package, device, audio, options);
        Assert.Equal(expected.Text, actual.Text);
        Assert.Equal(expected.Segments.SelectMany(s => s.Tokens), actual.Segments.SelectMany(s => s.Tokens));
    }

    /// <summary>Precompiling a speech package for the Vulkan device stores the kernels of both components; a load then compiles nothing new.</summary>
    [Fact]
    public async Task PrecompiledPackageLoadsWithStoredKernels()
    {
        string source = SpeechEnvironment.WhisperCpp("ggml-tiny.en");
        if (!SpeechEnvironment.Available(source, "vulkan"))
        {
            Assert.Skip("ggml-tiny.en, jfk.wav or a Vulkan device is missing.");
        }

        string package = Path.Combine(_directory, "tiny.en.slm");
        ModelConverter.Convert(source, package);
        var logs = new List<string>();
        var vulkan = new VulkanProvider();
        await PackagePrecompiler.PrecompileAsync(package, vulkan, vulkan.GetDevices()[0], new PrecompileOptions { Log = logs.Add },
            cancellationToken: TestContext.Current.CancellationToken);
        Assert.Contains(logs, l => l.Contains("Component 'encoder'", StringComparison.Ordinal));
        Assert.Contains(logs, l => l.Contains("Component 'decoder'", StringComparison.Ordinal));
        Assert.NotEmpty(ModelPackageReader.Open(package).ArtifactSets);

        var result = await Transcribe(package, "vulkan", WavFile.Read(SpeechEnvironment.Jfk), SpeechRecognitionOptions.Default);
        Assert.Contains("ask not what your country can do for you", result.Text, StringComparison.Ordinal);
    }

    /// <summary>Removes the scratch directory.</summary>
    public void Dispose()
    {
        if (Directory.Exists(_directory))
        {
            Directory.Delete(_directory, recursive: true);
        }
    }

    /// <summary>Transcribes through the service container, as an application does.</summary>
    private static async Task<Transcription> Transcribe(string path, string device, AudioBuffer audio, SpeechRecognitionOptions options)
    {
        var services = new ServiceCollection().AddSharplight()
            .AddSpeechRecognizer(o =>
            {
                o.ModelPath = path;
                o.Devices = [new DeviceSlot { Device = SpeechEnvironment.Device(device)! }];
            });
        await using var provider = services.BuildServiceProvider();
        return await provider.GetRequiredService<ISpeechRecognizer>().TranscribeAsync(audio, options);
    }
}
