using System.Diagnostics;
using Microsoft.AspNetCore.Http;
using Sharplight.Llm.Serving;

namespace Sharplight.AspNetCore;

/// <summary>Per-request state of the Sharplight pipeline: the resolved context, the endpoint options, the usage so far and admission handles.</summary>
internal sealed class SharplightRequestFeature(SharplightRequestContext context, SharplightEndpointOptions endpoint)
{
    private readonly long _started = Stopwatch.GetTimestamp();
    private readonly List<IAsyncDisposable> _handles = [];
    private long _firstToken;

    public SharplightRequestContext Context { get; } = context;
    public SharplightEndpointOptions Endpoint { get; } = endpoint;
    public IRequestAdmission? Admission { get; init; }

    /// <summary>The model name the request asked for (or the default alias), noted before the body is parsed.</summary>
    public string? RequestedModel { get; set; }

    /// <summary>The model name the client used, once known.</summary>
    public string? Model { get; set; }

    public long InputTokens { get; private set; }
    public long CachedInputTokens { get; private set; }
    public long OutputTokens { get; private set; }
    public bool Streaming { get; private set; }

    public TimeSpan Elapsed => Stopwatch.GetElapsedTime(_started);

    public TimeSpan? TimeToFirstToken => _firstToken == 0 ? null : Stopwatch.GetElapsedTime(_started, _firstToken);

    public static SharplightRequestFeature? Of(HttpContext http) => http.Features.Get<SharplightRequestFeature>();

    /// <summary>Adds the usage of a finished generation (prompt once per request, completions of every choice).</summary>
    public void AddUsage(GenerationUsage usage)
    {
        InputTokens = Math.Max(InputTokens, usage.PromptTokens);
        CachedInputTokens = Math.Max(CachedInputTokens, usage.CachedTokens);
        OutputTokens += usage.CompletionTokens;
    }

    /// <summary>Adds input tokens of a non-generative request (embedding, rerank).</summary>
    public void AddInput(long tokens) => InputTokens += tokens;

    /// <summary>Records the first streamed token (idempotent).</summary>
    public void MarkFirstToken()
    {
        Streaming = true;
        if (_firstToken == 0)
        {
            _firstToken = Stopwatch.GetTimestamp();
        }
    }

    public void MarkStreaming() => Streaming = true;

    public void Hold(IAsyncDisposable handle) => _handles.Add(handle);

    public async ValueTask ReleaseAsync()
    {
        foreach (var handle in _handles)
        {
            await handle.DisposeAsync();
        }

        _handles.Clear();
    }
}
