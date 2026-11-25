using System.Net;
using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Sharplight.AspNetCore;
using Sharplight.Cpu;
using Sharplight.Runtime;

namespace Sharplight.Serving.Tests;

/// <summary>
/// <c>/v1/audio/transcriptions</c> over a real Hugging Face speech model registered with <c>AddSpeechRecognizer</c> and served lazily
/// (skipped when the checkpoint or jfk.wav is absent): Cohere Transcribe on the CPU.
/// </summary>
[Collection("RealModel")]
public sealed class SpeechRecognitionRealTests
{
    private static readonly string s_models = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models");

    [Fact]
    public async Task Transcriptions_RunCohereTranscribe()
    {
        string model = Path.Combine(s_models, "hf", "cohere-transcribe-03-2026");
        string jfk = Path.Combine(s_models, "audio", "jfk.wav");
        Assert.SkipUnless(Directory.Exists(model) && File.Exists(jfk), "cohere-transcribe-03-2026 and jfk.wav not found");
        var ct = TestContext.Current.CancellationToken;
        var cpu = new CpuExecutionProvider();
        using var device = cpu.OpenDevice(cpu.GetDevices()[0]);
        var builder = WebApplication.CreateBuilder();
        builder.WebHost.UseTestServer();
        builder.Logging.ClearProviders();
        builder.Services.AddSharplight().AddSpeechRecognizer(o =>
        {
            o.ModelPath = model;
            o.Devices = [new DeviceSlot { Device = device }];
        });
        builder.Services.AddSharplightSpeechRecognizer("cohere-transcribe");
        await using var app = builder.Build();
        app.MapSharplightAudio();
        await app.StartAsync(ct);
        using var client = app.GetTestClient();

        using var form = new MultipartFormDataContent
        {
            { new ByteArrayContent(await File.ReadAllBytesAsync(jfk, ct)), "file", "jfk.wav" },
            { new StringContent("cohere-transcribe"), "model" },
            { new StringContent("en"), "language" },
            { new StringContent("verbose_json"), "response_format" },
        };
        var response = await client.PostAsync("/v1/audio/transcriptions", form, ct);
        string body = await response.Content.ReadAsStringAsync(ct);
        Assert.True(response.StatusCode == HttpStatusCode.OK, body);
        using var json = JsonDocument.Parse(body);
        Assert.Equal("And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.",
            json.RootElement.GetProperty("text").GetString());
        Assert.Equal("en", json.RootElement.GetProperty("language").GetString());
        Assert.Equal(1, json.RootElement.GetProperty("segments").GetArrayLength());
        await app.StopAsync(ct);
    }
}
