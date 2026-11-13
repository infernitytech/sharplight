using System.Buffers;
using System.IO.Pipelines;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Http;

namespace Sharplight.AspNetCore;

/// <summary>
/// Server-sent events writer over the response pipe. Events are serialized straight into the pipe (no intermediate strings)
/// and only reach the socket on <see cref="FlushAsync"/>, so a caller streaming several ready events flushes once.
/// </summary>
internal sealed class SseWriter
{
    private static readonly byte[] DataPrefix = "data: "u8.ToArray();
    private static readonly byte[] EventPrefix = "event: "u8.ToArray();
    private static readonly byte[] LineEnd = "\n"u8.ToArray();
    private static readonly byte[] EventEnd = "\n\n"u8.ToArray();
    private static readonly JsonWriterOptions WriterOptions = new() { Encoder = HttpJson.Options.Encoder, SkipValidation = true };

    private readonly HttpContext _http;
    private readonly PipeWriter _pipe;
    private readonly Utf8JsonWriter _json;

    public SseWriter(HttpContext http)
    {
        _http = http;
        http.Response.StatusCode = 200;
        http.Response.ContentType = "text/event-stream; charset=utf-8";
        http.Response.Headers.CacheControl = "no-cache";
        http.Response.Headers["X-Accel-Buffering"] = "no";
        _pipe = http.Response.BodyWriter;
        _json = new Utf8JsonWriter(_pipe, WriterOptions);
    }

    /// <summary>Starts a <c>data:</c> event (optionally named) and returns the JSON writer for its payload; finish with <see cref="EndEvent"/>.</summary>
    public Utf8JsonWriter BeginEvent(string? name = null)
    {
        if (name is not null)
        {
            _pipe.Write(EventPrefix);
            WriteAscii(name);
            _pipe.Write(LineEnd);
        }

        _pipe.Write(DataPrefix);
        _json.Reset(_pipe);
        return _json;
    }

    /// <summary>Completes the event started by <see cref="BeginEvent"/>.</summary>
    public void EndEvent()
    {
        _json.Flush();
        _pipe.Write(EventEnd);
    }

    /// <summary>Writes a <c>data:</c> event with <paramref name="node"/> as payload (buffered until <see cref="FlushAsync"/>).</summary>
    public void Data(JsonNode node)
    {
        node.WriteTo(BeginEvent(), HttpJson.Options);
        EndEvent();
    }

    /// <summary>Writes a named event with <paramref name="node"/> as payload (buffered until <see cref="FlushAsync"/>).</summary>
    public void Event(string name, JsonNode node)
    {
        node.WriteTo(BeginEvent(name), HttpJson.Options);
        EndEvent();
    }

    /// <summary>Writes raw text (buffered until <see cref="FlushAsync"/>).</summary>
    public void Raw(string text)
    {
        var span = _pipe.GetSpan(Encoding.UTF8.GetMaxByteCount(text.Length));
        _pipe.Advance(Encoding.UTF8.GetBytes(text, span));
    }

    /// <summary>Sends everything written so far.</summary>
    public async ValueTask FlushAsync() => await _pipe.FlushAsync(_http.RequestAborted);

    /// <summary>Writes a <c>data:</c> event and flushes it.</summary>
    public async Task DataAsync(JsonNode node)
    {
        Data(node);
        await FlushAsync();
    }

    /// <summary>Writes a named event and flushes it.</summary>
    public async Task EventAsync(string name, JsonNode node)
    {
        Event(name, node);
        await FlushAsync();
    }

    /// <summary>Writes raw text and flushes it.</summary>
    public async Task RawAsync(string text)
    {
        Raw(text);
        await FlushAsync();
    }

    private void WriteAscii(string text)
    {
        var span = _pipe.GetSpan(text.Length);
        int n = Encoding.ASCII.GetBytes(text, span);
        _pipe.Advance(n);
    }
}
