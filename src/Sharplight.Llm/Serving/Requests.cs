using System.Threading.Channels;

namespace Sharplight.Llm.Serving;

public enum FinishReason { Stop, Length, Cancelled, Error, Timeout }

public readonly record struct GenerationUsage(int PromptTokens, int CompletionTokens, int CachedTokens)
{
    public int TotalTokens => PromptTokens + CompletionTokens;
}

/// <summary>A token-level generation request.</summary>
public sealed class GenerationRequest
{
    public required int[] PromptTokens { get; init; }
    public SamplingParams Sampling { get; init; } = new();
    public int MaxTokens { get; init; } = 256;
    /// <summary>Extra token ids that end generation (in addition to the end-of-generation tokens when <see cref="StopAtEog"/>).</summary>
    public IReadOnlyList<int> StopTokenIds { get; init; } = [];
    public bool StopAtEog { get; init; } = true;
    public IReadOnlyList<string> StopStrings { get; init; } = [];
    /// <summary>Higher runs first; equal priorities are served first-come first-served.</summary>
    public int Priority { get; init; }
    /// <summary>Total time budget from submission (queue + generation). Exceeding it finishes with <see cref="FinishReason.Timeout"/>.</summary>
    public TimeSpan? Timeout { get; init; }
    /// <summary>Maximum time to wait for a slot; exceeding it finishes with <see cref="FinishReason.Timeout"/> without generating.</summary>
    public TimeSpan? QueueTimeout { get; init; }
    /// <summary>Report the log-probability of every sampled token (and <see cref="TopLogprobs"/> alternatives).</summary>
    public bool Logprobs { get; init; }
    public int TopLogprobs { get; init; }
    /// <summary>Hide control tokens from the generated text.</summary>
    public bool SkipSpecialTokens { get; init; }
}

/// <summary>
/// One streamed piece of a generation. Content chunks carry <see cref="Text"/>; the last chunk has <see cref="Finish"/> and
/// <see cref="Usage"/> set (and <see cref="Error"/> when it ended in an error).
/// </summary>
public sealed record GenerationChunk(
    string Text,
    IReadOnlyList<TokenLogprob>? Logprobs = null,
    FinishReason? Finish = null,
    GenerationUsage? Usage = null,
    string? Error = null,
    string? StopSequence = null);

public sealed class GenerationResult
{
    public string Text { get; init; } = "";
    public FinishReason Finish { get; init; }
    public GenerationUsage Usage { get; init; }
    public string? Error { get; init; }
    public string? StopSequence { get; init; }
    public IReadOnlyList<TokenLogprob> Logprobs { get; init; } = [];
}

/// <summary>Handle of a submitted request: stream its chunks, or cancel it.</summary>
public sealed class GenerationHandle
{
    internal GenerationHandle(long id, Channel<GenerationChunk> channel, Action cancel)
    {
        Id = id; _channel = channel; _cancel = cancel;
    }

    private readonly Channel<GenerationChunk> _channel;
    private readonly Action _cancel;
    public long Id { get; }

    public void Cancel() => _cancel();

    /// <summary>Streams the chunks until the final one. Abandoning the enumeration cancels the request.</summary>
    public async IAsyncEnumerable<GenerationChunk> ReadAllAsync([System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default)
    {
        bool done = false;
        try
        {
            while (await _channel.Reader.WaitToReadAsync(ct).ConfigureAwait(false))
            {
                while (_channel.Reader.TryRead(out var c))
                {
                    if (c.Finish is not null)
                    {
                        done = true;
                    }

                    yield return c;
                    if (done)
                    {
                        yield break;
                    }
                }
            }
        }
        finally { if (!done)
            {
                _cancel();
            }
        }
    }

    /// <summary>Waits for the whole generation and returns the concatenated text.</summary>
    public async Task<GenerationResult> CompleteAsync(CancellationToken ct = default)
    {
        var sb = new System.Text.StringBuilder();
        var lps = new List<TokenLogprob>();
        await foreach (var c in ReadAllAsync(ct).ConfigureAwait(false))
        {
            sb.Append(c.Text);
            if (c.Logprobs is not null)
            {
                lps.AddRange(c.Logprobs);
            }

            if (c.Finish is { } f)
            {
                return new GenerationResult { Text = sb.ToString(), Finish = f, Usage = c.Usage ?? default, Error = c.Error, StopSequence = c.StopSequence, Logprobs = lps };
            }
        }
        throw new InvalidOperationException("The generation ended without a final chunk.");
    }
}

/// <summary>Base of the exceptions thrown when a request is not admitted.</summary>
public class EngineException(string message, string code) : Exception(message)
{
    /// <summary>Machine-readable reason: <c>queue_full</c>, <c>context_length_exceeded</c>, <c>shutting_down</c>, <c>invalid_request</c>.</summary>
    public string Code { get; } = code;
}
public sealed class QueueFullException(int capacity) : EngineException($"The request queue is full ({capacity} waiting requests).", "queue_full");
public sealed class PromptTooLongException(int tokens, int max) : EngineException($"The prompt has {tokens} tokens, but the model context holds at most {max} (a prompt needs room for at least one generated token).", "context_length_exceeded")
{
    public int PromptTokens { get; } = tokens;
    public int MaxTokens { get; } = max;
}
public sealed class EngineStoppedException() : EngineException("The engine is shutting down and accepts no new requests.", "shutting_down");
