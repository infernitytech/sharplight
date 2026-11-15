using System.Net;
using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Sharplight.AspNetCore;
using Sharplight.Audio;

namespace Sharplight.Serving.Tests;

/// <summary>The OpenAI-compatible audio endpoints (<c>/v1/audio/transcriptions</c>, <c>/v1/audio/translations</c>) over a fake recognizer.</summary>
public sealed class AudioApiTests : IAsyncLifetime
{
    private readonly FakeSpeechRecognizer _recognizer = new();
    private WebApplication _app = null!;
    private HttpClient _client = null!;

    public async ValueTask InitializeAsync()
    {
        var builder = WebApplication.CreateBuilder();
        builder.WebHost.UseTestServer();
        builder.Logging.ClearProviders();
        builder.Services.AddSharplightSpeechRecognizer(_recognizer, "whisper-1");
        _app = builder.Build();
        _app.MapSharplightAudio();
        await _app.StartAsync();
        _client = _app.GetTestClient();
    }

    public async ValueTask DisposeAsync()
    {
        await _app.StopAsync();
        await _app.DisposeAsync();
    }

    [Fact]
    public async Task Transcription_ReturnsJsonText()
    {
        var response = await Post("/v1/audio/transcriptions", ("model", "whisper-1"), ("language", "de"));
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        using var body = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        Assert.Equal("Hello world.", body.RootElement.GetProperty("text").GetString());
        Assert.Equal("de", _recognizer.LastOptions!.Language);
        Assert.False(_recognizer.LastOptions.Translate);
        Assert.Equal(16000, _recognizer.LastAudio!.FrameCount);
    }

    [Fact]
    public async Task Translation_SetsTheTaskAndVerboseJsonListsSegments()
    {
        var response = await Post("/v1/audio/translations", ("model", "any-name"), ("response_format", "verbose_json"));
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        using var body = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        Assert.Equal("translate", body.RootElement.GetProperty("task").GetString());
        Assert.Equal(1.0, body.RootElement.GetProperty("duration").GetDouble(), 3);
        var segments = body.RootElement.GetProperty("segments");
        Assert.Equal(2, segments.GetArrayLength());
        Assert.Equal(0.5, segments[1].GetProperty("start").GetDouble(), 3);
        Assert.Equal(" translated.", segments[1].GetProperty("text").GetString());
        Assert.Equal(-0.25, segments[1].GetProperty("avg_logprob").GetDouble());
        Assert.True(_recognizer.LastOptions!.Translate);
    }

    [Fact]
    public async Task Subtitles_AreSrtAndVtt()
    {
        string srt = await (await Post("/v1/audio/transcriptions", ("response_format", "srt"))).Content.ReadAsStringAsync();
        Assert.StartsWith("1\n00:00:00,000 --> 00:00:00,500\nHello\n\n2\n", srt, StringComparison.Ordinal);
        string vtt = await (await Post("/v1/audio/transcriptions", ("response_format", "vtt"))).Content.ReadAsStringAsync();
        Assert.StartsWith("WEBVTT\n\n00:00:00.000 --> 00:00:00.500\nHello\n", vtt, StringComparison.Ordinal);
        string text = await (await Post("/v1/audio/transcriptions", ("response_format", "text"))).Content.ReadAsStringAsync();
        Assert.Equal("Hello world.\n", text);
    }

    [Fact]
    public async Task MissingFileOrBadFormat_IsBadRequest()
    {
        using var empty = new MultipartFormDataContent { { new StringContent("whisper-1"), "model" } };
        var missing = await _client.PostAsync("/v1/audio/transcriptions", empty);
        Assert.Equal(HttpStatusCode.BadRequest, missing.StatusCode);
        var badFormat = await Post("/v1/audio/transcriptions", ("response_format", "mp3"));
        Assert.Equal(HttpStatusCode.BadRequest, badFormat.StatusCode);
    }

    private async Task<HttpResponseMessage> Post(string path, params (string Name, string Value)[] fields)
    {
        var wav = new MemoryStream();
        WavFile.Write(wav, new AudioBuffer(new float[16000], 16000));
        var form = new MultipartFormDataContent { { new ByteArrayContent(wav.ToArray()), "file", "audio.wav" } };
        foreach (var (name, value) in fields)
        {
            form.Add(new StringContent(value), name);
        }

        return await _client.PostAsync(path, form);
    }
}
