using Microsoft.Extensions.DependencyInjection;
using Sharplight.Audio;
using Sharplight.Runtime;
using Xunit;

namespace Sharplight.Speech.Tests;

/// <summary>
/// The transcribe.cpp Parakeet GGUF of parakeet-tdt-0.6b-v3 (F16) against the transformers fixture of the Hugging Face checkpoint
/// (<c>parakeet-tdt-0.6b-v3.json</c>, see <c>ParakeetParityTests</c>): token-exact greedy TDT decoding on jfk.wav and a 29 s LibriSpeech
/// recording, CPU and Vulkan. The quantized files give the same text (checked by hand for Q8_0).
/// </summary>
public sealed class ParakeetGgufParityTests
{
    private const string File = "parakeet-tdt-0.6b-v3-F16.gguf";
    private const string Fixture = "parakeet-tdt-0.6b-v3.json";

    /// <summary>The recordings and devices.</summary>
    public static TheoryData<string, string> Cases()
    {
        var cases = new TheoryData<string, string>();
        foreach (string clip in new[] { SpeechEnvironment.JfkName, SpeechEnvironment.LibriSpeechName })
        {
            cases.Add(clip, "cpu");
            cases.Add(clip, "vulkan");
        }

        return cases;
    }

    /// <summary>Greedy decoding of the GGUF gives transformers' tokens and text.</summary>
    [Theory]
    [MemberData(nameof(Cases))]
    public async Task GreedyTokensMatchTransformers(string clip, string device)
    {
        string path = SpeechEnvironment.SpeechGguf(File);
        if (!SpeechEnvironment.Available(path, device) || !System.IO.File.Exists(SpeechEnvironment.Audio(clip)))
        {
            Assert.Skip($"{path}, {clip}.wav or the {device} device is missing.");
        }

        using var fixture = SpeechEnvironment.Fixture(Fixture);
        var expected = fixture.RootElement.GetProperty(clip);
        var services = new ServiceCollection().AddSharplight()
            .AddSpeechRecognizer(o =>
            {
                o.ModelPath = path;
                o.Devices = [new DeviceSlot { Device = SpeechEnvironment.Device(device)! }];
            });
        await using var provider = services.BuildServiceProvider();
        var recognizer = provider.GetRequiredService<ISpeechRecognizer>();
        var result = await recognizer.TranscribeAsync(WavFile.Read(SpeechEnvironment.Audio(clip)), new SpeechRecognitionOptions(), TestContext.Current.CancellationToken);
        TestContext.Current.TestOutputHelper?.WriteLine($"{clip} {device}: {result.Text}");
        int blank = 8192;
        int[] ids = [.. expected.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).Skip(1).Where(id => id != blank)];
        Assert.Equal(ids, result.Segments.SelectMany(s => s.Tokens).ToArray());
        Assert.Equal(expected.GetProperty("text").GetString()!.Trim(), result.Text);
    }
}
