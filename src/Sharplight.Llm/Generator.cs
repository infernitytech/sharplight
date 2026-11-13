using System.Diagnostics;
using Sharplight.Tokenizers;

namespace Sharplight.Llm;

public sealed record GenerationOptions
{
    public int MaxNewTokens { get; init; } = 256;
    public SamplingOptions Sampling { get; init; } = new();
    /// <summary>Stop on end-of-generation tokens.</summary>
    public bool StopAtEog { get; init; } = true;
    public IReadOnlyList<string> StopStrings { get; init; } = [];
}

public readonly record struct GeneratedToken(int Id, string Text);

public sealed record GenerationStats(int PromptTokens, int GeneratedTokens, TimeSpan PrefillTime, TimeSpan DecodeTime)
{
    public double PrefillTokensPerSecond => PromptTokens / Math.Max(1e-9, PrefillTime.TotalSeconds);
    public double DecodeTokensPerSecond => Math.Max(0, GeneratedTokens - 1) / Math.Max(1e-9, DecodeTime.TotalSeconds);
}

/// <summary>Autoregressive text generation on top of an <see cref="InferenceSession"/>.</summary>
public sealed class Generator(InferenceSession session, ITokenizer tokenizer)
{
    public GenerationStats? LastStats { get; private set; }

    /// <summary>Continues the session's context with <paramref name="prompt"/> and streams generated text.</summary>
    public IEnumerable<GeneratedToken> Generate(string prompt, GenerationOptions options, bool parseSpecial = true)
        => Generate(tokenizer.Encode(prompt, addSpecial: session.Position == 0, parseSpecial: parseSpecial), options);

    public IEnumerable<GeneratedToken> Generate(int[] promptTokens, GenerationOptions options)
    {
        var sampler = new Sampler(options.Sampling);
        foreach (var t in promptTokens) sampler.Accept(t);
        var decoder = new StreamingDecoder(tokenizer);
        var text = new System.Text.StringBuilder();

        var sw = Stopwatch.StartNew();
        var logits = session.Process(promptTokens);
        var prefill = sw.Elapsed;
        sw.Restart();
        int generated = 0;
        try
        {
            for (int i = 0; i < options.MaxNewTokens; i++)
            {
                int id = sampler.Sample(logits);
                sampler.Accept(id);
                generated++;
                if (options.StopAtEog && tokenizer.IsEndOfGeneration(id)) break;
                var piece = decoder.Push(id);
                text.Append(piece);
                yield return new GeneratedToken(id, piece);
                if (options.StopStrings.Any(s => text.ToString().EndsWith(s, StringComparison.Ordinal))) break;
                if (i + 1 < options.MaxNewTokens) logits = session.Process([id]);
            }
            var tail = decoder.Flush();
            if (tail.Length > 0) yield return new GeneratedToken(-1, tail);
        }
        finally
        {
            LastStats = new GenerationStats(promptTokens.Length, generated, prefill, sw.Elapsed);
        }
    }
}
