using System.Text;
using System.Threading.Channels;

namespace Sharplight.Llm.Serving.Chat;

/// <summary>A submitted chat/completion request: stream <see cref="ChatEvent"/>s or wait for the <see cref="ChatResult"/>.</summary>
public sealed class ChatRun
{
    private readonly IReadOnlyList<GenerationHandle> _handles;
    private readonly Func<ChatOutputParser>? _parserFactory;

    internal ChatRun(IReadOnlyList<GenerationHandle> handles, Func<ChatOutputParser>? parserFactory, int promptTokens)
    {
        _handles = handles;
        _parserFactory = parserFactory;
        PromptTokens = promptTokens;
    }

    public int PromptTokens { get; }
    public int ChoiceCount => _handles.Count;

    public void Cancel()
    {
        foreach (var h in _handles)
        {
            h.Cancel();
        }
    }

    /// <summary>Streams events of all choices (interleaved when N &gt; 1). Abandoning the enumeration cancels the requests.</summary>
    public async IAsyncEnumerable<ChatEvent> ReadAllAsync([System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default)
    {
        if (_handles.Count == 1)
        {
            await foreach (var e in Choice(0, ct).ConfigureAwait(false))
            {
                yield return e;
            }

            yield break;
        }
        var ch = Channel.CreateUnbounded<ChatEvent>(new UnboundedChannelOptions { SingleReader = true });
        var tasks = Enumerable.Range(0, _handles.Count).Select(i => Task.Run(async () =>
        {
            await foreach (var e in Choice(i, ct).ConfigureAwait(false))
            {
                ch.Writer.TryWrite(e);
            }
        }, CancellationToken.None)).ToArray();
        _ = Task.WhenAll(tasks).ContinueWith(t => ch.Writer.TryComplete(t.Exception?.GetBaseException()), TaskScheduler.Default);
        try
        {
            await foreach (var e in ch.Reader.ReadAllAsync(ct).ConfigureAwait(false))
            {
                yield return e;
            }
        }
        finally { Cancel(); }
    }

    private async IAsyncEnumerable<ChatEvent> Choice(int index, [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct)
    {
        var parser = _parserFactory?.Invoke();
        await foreach (var c in _handles[index].ReadAllAsync(ct).ConfigureAwait(false))
        {
            if (c.Text.Length > 0)
            {
                if (parser is null)
                {
                    yield return new ChatEvent(index, Content: c.Text);
                }
                else
                {
                    foreach (var e in Map(index, parser.Push(c.Text)))
                    {
                        yield return e;
                    }
                }
            }
            if (c.Logprobs is not null)
            {
                yield return new ChatEvent(index, Logprobs: c.Logprobs);
            }

            if (c.Finish is { } f)
            {
                if (parser is not null)
                {
                    foreach (var e in Map(index, parser.Finish()))
                    {
                        yield return e;
                    }
                }

                var reason = f switch
                {
                    FinishReason.Stop => parser?.ToolCallCount > 0 ? ChatFinishReason.ToolCalls : ChatFinishReason.Stop,
                    FinishReason.Length => ChatFinishReason.Length,
                    FinishReason.Cancelled => ChatFinishReason.Cancelled,
                    FinishReason.Timeout => ChatFinishReason.Timeout,
                    _ => ChatFinishReason.Error,
                };
                yield return new ChatEvent(index, Finish: reason, Usage: c.Usage, StopSequence: c.StopSequence, Error: c.Error);
            }
        }
    }

    private static IEnumerable<ChatEvent> Map(int index, List<ParsedPiece> pieces)
    {
        foreach (var p in pieces)
        {
            yield return p.Kind switch
            {
                PieceKind.Content => new ChatEvent(index, Content: p.Text),
                PieceKind.Reasoning => new ChatEvent(index, Reasoning: p.Text),
                _ => new ChatEvent(index, ToolCall: p.Call),
            };
        }
    }

    public async Task<ChatResult> CompleteAsync(CancellationToken ct = default)
    {
        int n = _handles.Count;
        var content = Enumerable.Range(0, n).Select(_ => new StringBuilder()).ToArray();
        var reasoning = Enumerable.Range(0, n).Select(_ => new StringBuilder()).ToArray();
        var calls = Enumerable.Range(0, n).Select(_ => new List<ToolCall>()).ToArray();
        var lps = Enumerable.Range(0, n).Select(_ => new List<TokenLogprob>()).ToArray();
        var choices = new ChatChoice?[n];
        await foreach (var e in ReadAllAsync(ct).ConfigureAwait(false))
        {
            int i = e.Index;
            if (e.Content is not null)
            {
                content[i].Append(e.Content);
            }

            if (e.Reasoning is not null)
            {
                reasoning[i].Append(e.Reasoning);
            }

            if (e.ToolCall is not null)
            {
                calls[i].Add(e.ToolCall);
            }

            if (e.Logprobs is not null)
            {
                lps[i].AddRange(e.Logprobs);
            }

            if (e.Finish is { } f)
            {
                choices[i] = new ChatChoice
                {
                    Index = i, Content = content[i].ToString(), Reasoning = reasoning[i].Length > 0 ? reasoning[i].ToString() : null,
                    ToolCalls = calls[i], Finish = f, StopSequence = e.StopSequence, Error = e.Error, Logprobs = lps[i], Usage = e.Usage ?? default,
                };
            }
        }
        var done = choices.Select(c => c ?? throw new InvalidOperationException("A choice ended without a finish event.")).ToList();
        return new ChatResult
        {
            Choices = done,
            Usage = new GenerationUsage(done[0].Usage.PromptTokens, done.Sum(c => c.Usage.CompletionTokens), done[0].Usage.CachedTokens),
        };
    }
}
