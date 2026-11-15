using System.Drawing;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Sharplight.Audio;
using Sharplight.Imaging;
using Sharplight.Speech;
using Sharplight.Vision;
using MeaiImageGenerationOptions = Microsoft.Extensions.AI.ImageGenerationOptions;
using MeaiImageGenerationRequest = Microsoft.Extensions.AI.ImageGenerationRequest;

namespace Sharplight.Extensions.AI.Tests;

/// <summary>The Microsoft.Extensions.AI speech and image abstractions over the Sharplight task clients.</summary>
public class TaskClientAdapterTests
{
    [Fact]
    public async Task SpeechToTextReadsWaveAndMapsSegmentsAndOptions()
    {
        var recognizer = new FakeRecognizer();
        using var client = new SharplightSpeechToTextClient(recognizer, ownsRecognizer: true);
        var cancellation = TestContext.Current.CancellationToken;

        var response = await client.GetTextAsync(Wave(new AudioBuffer(new float[4800 * 2], 24000, 2)),
            new SpeechToTextOptions { SpeechLanguage = "de", TextLanguage = "en" }, cancellation);
        Assert.Equal("[0][100]", response.Text);
        Assert.Equal(TimeSpan.FromMilliseconds(200), response.EndTime);
        Assert.Equal("fake-whisper", response.ModelId);
        Assert.Equal((24000, 2), (recognizer.LastAudio!.SampleRate, recognizer.LastAudio.Channels));
        Assert.Equal("de", recognizer.LastOptions!.Language);
        Assert.True(recognizer.LastOptions.Translate);

        var updates = new List<SpeechToTextResponseUpdate>();
        await foreach (var update in client.GetStreamingTextAsync(Wave(new AudioBuffer(new float[1600], 16000)), cancellationToken: cancellation))
        {
            updates.Add(update);
        }

        var single = Assert.Single(updates);
        Assert.Equal(SpeechToTextResponseUpdateKind.TextUpdated, single.Kind);
        Assert.Equal((TimeSpan.Zero, TimeSpan.FromMilliseconds(100)), (single.StartTime!.Value, single.EndTime!.Value));
        Assert.False(recognizer.LastOptions.Translate);

        Assert.Equal("sharplight", client.GetService<SpeechToTextClientMetadata>()!.ProviderName);
        Assert.Same(recognizer, client.GetService<ISpeechRecognizer>());
        client.Dispose();
        Assert.True(recognizer.Disposed);
    }

    [Fact]
    public async Task TextToSpeechReturnsWaveAndMapsVoiceAndSpeed()
    {
        var synthesizer = new FakeSynthesizer();
        using var client = new SharplightTextToSpeechClient(synthesizer);
        var cancellation = TestContext.Current.CancellationToken;

        var response = await client.GetAudioAsync("abc", new TextToSpeechOptions { VoiceId = "bob", Speed = 1.25f }, cancellation);
        var content = Assert.IsType<DataContent>(Assert.Single(response.Contents));
        Assert.Equal("audio/wav", content.MediaType);
        var audio = WavFile.Read(new MemoryStream(content.Data.ToArray()));
        Assert.Equal((8000, 240), (audio.SampleRate, audio.FrameCount));
        Assert.All(audio.Samples, s => Assert.Equal(0.5f, s, 3));
        Assert.Equal(("bob", 1.25f), (synthesizer.LastOptions!.Voice, synthesizer.LastOptions.Speed));

        int chunks = 0;
        await foreach (var update in client.GetStreamingAudioAsync("hi", cancellationToken: cancellation))
        {
            Assert.Equal(TextToSpeechResponseUpdateKind.AudioUpdating, update.Kind);
            chunks++;
        }

        Assert.Equal(2, chunks);
        await Assert.ThrowsAsync<NotSupportedException>(() => client.GetAudioAsync("x", new TextToSpeechOptions { AudioFormat = "mp3" }, cancellation));
    }

    [Fact]
    public async Task ImageGeneratorReturnsPngAndMapsSizeCountAndExtras()
    {
        var generator = new FakeImageGenerator();
        using var adapter = new SharplightImageGenerator(generator);
        var original = new ImageInput(2, 2, [.. Enumerable.Range(0, 12).Select(i => (byte)(i * 20))]);
        var request = new MeaiImageGenerationRequest("a red fox", [new DataContent(PngCodec.Encode(original), "image/png")]);
        var options = new MeaiImageGenerationOptions
        {
            Count = 2,
            ImageSize = new Size(16, 12),
            AdditionalProperties = new AdditionalPropertiesDictionary { ["seed"] = 40L, ["steps"] = 20, ["guidance_scale"] = 7.5, ["negative_prompt"] = "blurry" },
        };

        var response = await adapter.GenerateAsync(request, options, TestContext.Current.CancellationToken);
        Assert.Equal(2, response.Contents.Count);
        var second = PngCodec.Decode(Assert.IsType<DataContent>(response.Contents[1]).Data.Span);
        Assert.Equal((16, 12), (second.Width, second.Height));
        Assert.All(second.Rgb, b => Assert.Equal(41, b));

        var seen = generator.LastRequest!;
        Assert.Equal(("a red fox", "blurry"), (seen.Prompt, seen.NegativePrompt));
        Assert.Equal(original.Rgb, seen.Image!.Rgb);
        var seenOptions = generator.LastOptions!;
        Assert.Equal((20, 7.5f, 40L), (seenOptions.Steps!.Value, seenOptions.GuidanceScale!.Value, seenOptions.Seed!.Value));
        await Assert.ThrowsAsync<NotSupportedException>(() => adapter.GenerateAsync(request, new MeaiImageGenerationOptions { MediaType = "image/jpeg" },
            TestContext.Current.CancellationToken));
    }

    [Fact]
    public void AdaptersResolveOverTheContainersTaskClients()
    {
        var services = new ServiceCollection()
            .AddSingleton<ISpeechRecognizer, FakeRecognizer>()
            .AddKeyedSingleton<ISpeechSynthesizer, FakeSynthesizer>("tts")
            .AddSingleton<ImageGeneration.IImageGenerator, FakeImageGenerator>()
            .AddSharplightSpeechToTextClient()
            .AddSharplightTextToSpeechClient("tts")
            .AddSharplightImageGenerator();
        using var provider = services.BuildServiceProvider();
        Assert.Same(provider.GetRequiredService<ISpeechRecognizer>(), provider.GetRequiredService<ISpeechToTextClient>().GetService<ISpeechRecognizer>());
        Assert.IsType<SharplightTextToSpeechClient>(provider.GetRequiredKeyedService<ITextToSpeechClient>("tts"));
        Assert.IsType<SharplightImageGenerator>(provider.GetRequiredService<IImageGenerator>());
    }

    private static MemoryStream Wave(AudioBuffer audio)
    {
        var stream = new MemoryStream();
        WavFile.Write(stream, audio, WavSampleFormat.Float32);
        stream.Position = 0;
        return stream;
    }
}
