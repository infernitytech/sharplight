using System.Security.Cryptography;
using Sharplight.Audio;
using Sharplight.Llm;
using Sharplight.Llm.Serving;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Speech;
using Sharplight.Tasks;
using Sharplight.Vision;
using Xunit;

namespace Sharplight.Speech.Tests;

/// <summary>
/// Audio-language prompts through the batching engine: Qwen3-ASR prompts whose audio rows come from the model's audio tower, submitted together
/// as <see cref="GenerationRequest.PromptMedia"/> to an engine over a decoder with media input, give exactly the greedy tokens of the single-session
/// path the speech-recognition client runs (the multimodal graph fed the assembled prompt), on the CPU and on Vulkan; a repeated prompt takes its
/// audio rows' KV blocks from the prefix cache.
/// </summary>
public sealed class AudioLanguageBatchedTests
{
    private const int MaxTokens = 48;

    public static TheoryData<string> Devices() => ["cpu", "vulkan"];

    [Theory]
    [MemberData(nameof(Devices))]
    public async Task Qwen3AsrBatchedMatchesTheSession(string device)
    {
        string path = SpeechEnvironment.HuggingFace("Qwen3-ASR-0.6B");
        if (!SpeechEnvironment.Available(path, device) || !File.Exists(SpeechEnvironment.Audio(SpeechEnvironment.LibriSpeechName)))
        {
            Assert.Skip($"{path}, the recordings or the {device} device is missing.");
        }

        var ct = TestContext.Current.CancellationToken;
        using var model = TaskModel.Load(TestHost.Loader, new TaskLoadOptions { ModelPath = path, Devices = [new DeviceSlot { Device = SpeechEnvironment.Device(device)! }], ContextLength = 1024 });
        using var recognizer = new AudioLanguageSpeechRecognizer(model);
        var extractor = AudioFeatureExtractors.Create(model);
        using var llm = LlmModel.FromTaskModel(model);

        // Two recordings (the first 10 s of each), each with the language detected and with English forced.
        var prompts = new List<(int[] Tokens, float[] Embeddings, int Start)>();
        foreach (string clip in new[] { SpeechEnvironment.JfkName, SpeechEnvironment.LibriSpeechName })
        {
            var samples = WavFile.Read(SpeechEnvironment.Audio(clip)).ToModelInput(extractor.SampleRate).Samples;
            var features = extractor.Extract(samples);
            int frames = Math.Min(features.Frames, 10 * extractor.SampleRate / extractor.HopLength);
            var embeddings = recognizer.Encode(features, 0, frames);
            int rows = recognizer.AudioRows(frames);
            foreach (string? language in new[] { null, "en" })
            {
                var tokens = recognizer.PromptTokens(rows, new SpeechRecognitionOptions { Language = language });
                int start = RunStart(tokens, rows);
                prompts.Add((tokens, embeddings, start));
            }
        }

        int axes = llm.Definition.Property(WellKnown.PositionAxes, 1);
        var expected = prompts.Select(p => SessionGreedy(llm, p.Tokens, p.Embeddings, p.Start, axes)).ToList();

        using var engine = new InferenceEngine(llm.CreateBatchDecoder(slots: 4, mediaInput: true), llm.Tokenizer);
        var requests = prompts.Select(p => Request(p.Tokens, p.Embeddings, p.Start, p.Embeddings.Length / Rows(p.Tokens, p.Start), axes)).ToList();
        var handles = requests.Select(r => engine.Submit(r, ct)).ToList();
        var results = await Task.WhenAll(handles.Select(h => h.CompleteAsync(ct)));
        for (int i = 0; i < prompts.Count; i++)
        {
            Assert.Null(results[i].Error);
            var ids = results[i].Logprobs.Select(l => l.TokenId).ToArray();
            TestContext.Current.TestOutputHelper?.WriteLine($"{device} prompt {i}: {llm.Tokenizer.Decode(ids)}");
            Assert.Equal(expected[i], ids);
        }

        var again = await engine.Submit(requests[0], ct).CompleteAsync(ct);
        Assert.Equal(expected[0], again.Logprobs.Select(l => l.TokenId).ToArray());
        Assert.True(again.Usage.CachedTokens >= requests[0].PromptMedia[0].End, $"cached {again.Usage.CachedTokens} prompt tokens");
    }

    /// <summary>The session path of the recognizer: the prompt with its audio rows through the multimodal graph, then greedy tokens until the end of the turn.</summary>
    private static int[] SessionGreedy(LlmModel llm, int[] tokens, float[] embeddings, int start, int axes)
    {
        using var session = llm.CreateSession(WellKnown.DecodeMultimodalGraph);
        int rows = Rows(tokens, start);
        var prompt = new MultimodalPrompt
        {
            Tokens = tokens,
            ImageIndex = [.. tokens.Select((_, i) => i >= start && i < start + rows ? 0 : -1)],
            ImageRow = [.. tokens.Select((_, i) => i >= start && i < start + rows ? i - start : 0)],
            Positions = Positions(tokens.Length, axes),
            PositionAxes = axes,
            NextPosition = tokens.Length,
        };
        var ids = new List<int>();
        var logits = session.Process(prompt, [embeddings]);
        while (ids.Count < MaxTokens)
        {
            int next = Enumerable.Range(0, logits.Length).MaxBy(i => logits[i]);
            ids.Add(next);
            if (llm.Tokenizer.IsEndOfGeneration(next))
            {
                break;
            }

            logits = session.Process([next]);
        }

        return [.. ids];
    }

    private static GenerationRequest Request(int[] tokens, float[] embeddings, int start, int width, int axes) => new()
    {
        PromptTokens = tokens,
        PromptMedia = [new PromptMedia(start, new EncodedMedia(Hash(embeddings), embeddings, width))],
        PromptPositions = Positions(tokens.Length, axes),
        MaxTokens = MaxTokens,
        Logprobs = true,
        Sampling = SamplingParams.GreedyDefault,
    };

    /// <summary>Every row's position is its index, on every axis (the recognizer's prompt layout).</summary>
    private static int[] Positions(int rows, int axes) => [.. Enumerable.Range(0, rows * axes).Select(i => i / axes)];

    /// <summary>Where the run of <paramref name="rows"/> equal tokens (the audio placeholders) starts.</summary>
    private static int RunStart(int[] tokens, int rows)
    {
        for (int i = 0; i + rows <= tokens.Length; i++)
        {
            if (Rows(tokens, i) == rows)
            {
                return i;
            }
        }

        throw new InvalidOperationException("The prompt has no run of audio placeholders.");
    }

    private static int Rows(int[] tokens, int start)
    {
        int end = start;
        while (end < tokens.Length && tokens[end] == tokens[start])
        {
            end++;
        }

        return end - start;
    }

    private static UInt128 Hash(float[] rows)
    {
        Span<byte> digest = stackalloc byte[32];
        SHA256.HashData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(rows.AsSpan()), digest);
        return System.Runtime.InteropServices.MemoryMarshal.Read<UInt128>(digest);
    }
}
