using System.Threading.Channels;

namespace Sharplight.Llm.Serving;

/// <summary>Handle of a submitted request: stream its chunks, or cancel it.</summary>
public sealed class GenerationHandle
{
    internal GenerationHandle(long id, Channel<GenerationChunk> channel, Action cancel)
    {
        Id = id;
        _channel = channel;
        _cancel = cancel;
    }

    private readonly Channel<GenerationChunk> _channel;
    private readonly Action _cancel;
    /// <summary>Engine-wide unique, increasing request id.</summary>
    public long Id { get; }

    /// <summary>Asks the engine to stop the request; the stream then ends with <see cref="FinishReason.Cancelled"/>. Idempotent, thread-safe.</summary>
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
        finally
        {
            if (!done)
            {
                _cancel();
            }
        }
    }

    /// <summary>Waits for the whole generation and returns the concatenated text.</summary>
    /// <exception cref="OperationCanceledException"><paramref name="ct"/> was cancelled (the request is cancelled too).</exception>
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
