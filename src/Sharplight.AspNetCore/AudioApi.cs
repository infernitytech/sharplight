using System.Globalization;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Http.Features;
using Microsoft.AspNetCore.Routing;
using Microsoft.Extensions.DependencyInjection;
using Sharplight.Audio;
using Sharplight.Speech;

namespace Sharplight.AspNetCore;

/// <summary>
/// OpenAI-compatible audio endpoints: <c>POST /v1/audio/transcriptions</c> and <c>POST /v1/audio/translations</c> (multipart form with
/// <c>file</c> (WAV), <c>model</c>, <c>language</c>, <c>prompt</c>, <c>response_format</c> = <c>json</c> (default), <c>text</c>, <c>srt</c>,
/// <c>vtt</c> or <c>verbose_json</c>, and <c>temperature</c>). Recognizers are registered with <c>services.AddSharplightSpeechRecognizer(...)</c>;
/// <c>MapSharplightServer</c> maps the endpoints when any is registered, or call <see cref="MapSharplightAudio"/>. Requests to one recognizer run
/// one at a time (recognizers are not thread-safe; the recognizer queues them).
/// </summary>
/// <example>
/// <code>
/// builder.Services.AddSharplight().AddSpeechRecognizer(o => { o.ModelPath = "whisper-small.slm"; o.Devices = [slot]; });
/// builder.Services.AddSharplightSpeechRecognizer("whisper-1");   // serves the container's ISpeechRecognizer
/// var app = builder.Build();
/// app.MapSharplightAudio();   // curl -F file=@talk.wav -F model=whisper-1 http://host/v1/audio/transcriptions
/// </code>
/// </example>
public static class AudioApi
{
    extension(IEndpointRouteBuilder routes)
    {
        /// <summary>
        /// Maps <c>/v1/audio/transcriptions</c> and <c>/v1/audio/translations</c> (when a recognizer is registered or the server serves the host's models) and
        /// <c>/v1/audio/speech</c> (when a synthesizer is registered or the server serves the host's models, <see cref="SharplightServerOptions.ServeHostModels"/>).
        /// </summary>
        public IEndpointRouteBuilder MapSharplightAudio()
        {
            bool host = HostTaskModels.Enabled(routes.ServiceProvider);
            var registered = routes.ServiceProvider.GetService<AudioModelRegistrations>();
            if (registered is null && !host)
            {
                return routes;
            }

            if (registered is null || registered.Recognizers.Count + registered.Lazy.Count > 0 || host)
            {
                routes.MapPost("/v1/audio/transcriptions", http => Handle(http, translate: false));
                routes.MapPost("/v1/audio/translations", http => Handle(http, translate: true));
            }

            if (host || registered?.Synthesizers.Count > 0)
            {
                routes.MapPost("/v1/audio/speech", http => HandleSpeech(http));
            }

            return routes;
        }
    }

    private static async Task Handle(HttpContext http, bool translate)
    {
        try
        {
            await Recognize(http, translate);
        }
        catch (ApiException ex)
        {
            await ApiErrors.WriteAsync(http, ex);
        }
        catch (Exception ex) when (ex is ArgumentException or NotSupportedException or InvalidDataException)
        {
            await ApiErrors.WriteAsync(http, 400, "invalid_request_error", ex.Message);
        }
    }

    private static async Task Recognize(HttpContext http, bool translate)
    {
        var models = http.RequestServices.GetService<AudioModels>() ?? new AudioModels();
        if (http.Features.Get<IHttpMaxRequestBodySizeFeature>() is { IsReadOnly: false } limit)
        {
            limit.MaxRequestBodySize = models.MaxRequestBodyBytes;
        }

        if (!http.Request.HasFormContentType)
        {
            throw ApiException.BadRequest("The request must be multipart/form-data with a 'file' field.", "file");
        }

        var form = await http.Request.ReadFormAsync(http.RequestAborted);
        var file = form.Files.GetFile("file") ?? throw ApiException.BadRequest("Missing the 'file' field (a WAV file).", "file");
        using var leased = await Pick(http, models, form["model"].FirstOrDefault());
        var recognizer = leased.Client;
        string format = form["response_format"].FirstOrDefault() ?? "json";
        if (format is not ("json" or "text" or "srt" or "vtt" or "verbose_json"))
        {
            throw ApiException.BadRequest("'response_format' must be json, text, srt, vtt or verbose_json.", "response_format");
        }

        AudioBuffer audio;
        await using (var stream = file.OpenReadStream())
        {
            var buffer = new MemoryStream();
            await stream.CopyToAsync(buffer, http.RequestAborted);
            buffer.Position = 0;
            audio = WavFile.Read(buffer);
        }

        float temperature = float.TryParse(form["temperature"].FirstOrDefault(), NumberStyles.Float, CultureInfo.InvariantCulture, out float t) ? t : 0f;
        var options = new SpeechRecognitionOptions
        {
            Language = translate ? null : NullIfEmpty(form["language"].FirstOrDefault()),
            Translate = translate,
            Prompt = NullIfEmpty(form["prompt"].FirstOrDefault()),
            Sampling = temperature > 0 ? new Llm.SamplingOptions { Temperature = temperature, TopK = 0, TopP = 1 } : Llm.SamplingOptions.Greedy,
        };
        var result = await recognizer.TranscribeAsync(audio, options, http.RequestAborted);
        await Write(http, result, format, translate);
    }

    private static async Task<Leased<ISpeechRecognizer>> Pick(HttpContext http, AudioModels models, string? requested)
    {
        if (requested is not null && models.Recognizers.TryGetValue(requested, out var recognizer))
        {
            return new Leased<ISpeechRecognizer>(recognizer, null);
        }

        if (await HostTaskModels.TryAcquireAsync<ISpeechRecognizer>(http, requested, "speech recognition") is { } hosted)
        {
            return new Leased<ISpeechRecognizer>(hosted.Client, hosted.Lease);
        }

        // A single model answers to any name (clients often hard-code "whisper-1").
        if (models.Recognizers.Count > 0 && (requested is null || models.Recognizers.Count == 1))
        {
            return new Leased<ISpeechRecognizer>(models.Recognizers[models.DefaultRecognizer ?? models.Recognizers.Keys.First()], null);
        }

        throw new ApiException(404, "invalid_request_error", $"The speech model '{requested}' does not exist (available: {string.Join(", ", models.Recognizers.Keys)}).",
            "model_not_found", "model");
    }

    private static async Task HandleSpeech(HttpContext http)
    {
        try
        {
            await Speak(http);
        }
        catch (ApiException ex)
        {
            await ApiErrors.WriteAsync(http, ex);
        }
        catch (Exception ex) when (ex is ArgumentException or NotSupportedException)
        {
            await ApiErrors.WriteAsync(http, 400, "invalid_request_error", ex.Message);
        }
    }

    private static async Task Speak(HttpContext http)
    {
        var models = http.RequestServices.GetService<AudioModels>() ?? new AudioModels();
        using var document = await HttpJson.ReadBodyAsync(http, 1024 * 1024);
        var body = document.RootElement;
        string input = body.Str("input") is { Length: > 0 } text ? text : throw ApiException.BadRequest("'input' is required.", "input");
        string format = body.Str("response_format") ?? "wav";
        if (format is not ("wav" or "pcm"))
        {
            throw ApiException.BadRequest("'response_format' must be wav (the default) or pcm (16-bit little-endian mono at the model's sample rate).", "response_format");
        }

        string? requested = body.Str("model");
        ISpeechSynthesizer synthesizer;
        IDisposable? lease = null;
        if (requested is not null && models.Synthesizers.TryGetValue(requested, out var registered))
        {
            synthesizer = registered;
        }
        else if (await HostTaskModels.TryAcquireAsync<ISpeechSynthesizer>(http, requested, "speech synthesis") is { } hosted)
        {
            (synthesizer, lease) = hosted;
        }
        else if (models.Synthesizers.Count > 0 && (requested is null || models.Synthesizers.Count == 1))
        {
            synthesizer = models.Synthesizers[models.DefaultSynthesizer ?? models.Synthesizers.Keys.First()];
        }
        else
        {
            throw new ApiException(404, "invalid_request_error", $"The speech synthesis model '{requested}' does not exist.", "model_not_found", "model");
        }

        using (lease)
        {
            var options = new SpeechSynthesisOptions { Voice = body.Str("voice"), Speed = body.Float("speed") ?? 1f };
            if (format == "pcm")
            {
                http.Response.ContentType = "audio/L16; rate=" + synthesizer.SampleRate.ToString(CultureInfo.InvariantCulture) + "; channels=1";
                await foreach (var chunk in synthesizer.SynthesizeStreamingAsync(input, options, http.RequestAborted))
                {
                    var bytes = new byte[chunk.Samples.Length * 2];
                    for (int i = 0; i < chunk.Samples.Length; i++)
                    {
                        short sample = (short)Math.Clamp((int)MathF.Round(chunk.Samples[i] * short.MaxValue), short.MinValue, short.MaxValue);
                        bytes[2 * i] = (byte)sample;
                        bytes[2 * i + 1] = (byte)(sample >> 8);
                    }

                    await http.Response.Body.WriteAsync(bytes, http.RequestAborted);
                    await http.Response.Body.FlushAsync(http.RequestAborted);
                }

                return;
            }

            var result = await synthesizer.SynthesizeAsync(input, options, http.RequestAborted);
            using var wav = new MemoryStream();
            WavFile.Write(wav, result.Audio);
            http.Response.ContentType = "audio/wav";
            http.Response.ContentLength = wav.Length;
            await http.Response.Body.WriteAsync(wav.ToArray(), http.RequestAborted);
        }
    }

    private readonly record struct Leased<T>(T Client, IDisposable? Lease) : IDisposable
    {
        public void Dispose() => Lease?.Dispose();
    }

    private static async Task Write(HttpContext http, Transcription result, string format, bool translate)
    {
        switch (format)
        {
            case "text":
                http.Response.ContentType = "text/plain; charset=utf-8";
                await http.Response.WriteAsync(result.Text + "\n", http.RequestAborted);
                return;
            case "srt" or "vtt":
                http.Response.ContentType = format == "srt" ? "application/x-subrip; charset=utf-8" : "text/vtt; charset=utf-8";
                await http.Response.WriteAsync(Subtitles(result, format == "vtt"), http.RequestAborted);
                return;
        }

        http.Response.ContentType = "application/json";
        await using var writer = new Utf8JsonWriter(http.Response.Body);
        writer.WriteStartObject();
        if (format == "verbose_json")
        {
            writer.WriteString("task", translate ? "translate" : "transcribe");
            writer.WriteString("language", result.Language);
            writer.WriteNumber("duration", result.Duration.TotalSeconds);
        }

        writer.WriteString("text", result.Text);
        if (format == "verbose_json")
        {
            writer.WriteStartArray("segments");
            for (int i = 0; i < result.Segments.Count; i++)
            {
                var segment = result.Segments[i];
                writer.WriteStartObject();
                writer.WriteNumber("id", i);
                writer.WriteNumber("start", segment.Start.TotalSeconds);
                writer.WriteNumber("end", segment.End.TotalSeconds);
                writer.WriteString("text", segment.Text);
                writer.WriteStartArray("tokens");
                foreach (int token in segment.Tokens)
                {
                    writer.WriteNumberValue(token);
                }

                writer.WriteEndArray();
                if (segment.AverageLogProbability is double logProbability)
                {
                    writer.WriteNumber("avg_logprob", logProbability);
                }

                if (segment.NoSpeechProbability is double noSpeech)
                {
                    writer.WriteNumber("no_speech_prob", noSpeech);
                }

                writer.WriteEndObject();
            }

            writer.WriteEndArray();
        }

        writer.WriteEndObject();
        await writer.FlushAsync(http.RequestAborted);
    }

    private static string Subtitles(Transcription result, bool vtt)
    {
        var text = new StringBuilder(vtt ? "WEBVTT\n\n" : "");
        for (int i = 0; i < result.Segments.Count; i++)
        {
            var segment = result.Segments[i];
            if (!vtt)
            {
                text.Append(CultureInfo.InvariantCulture, $"{i + 1}\n");
            }

            text.Append(CultureInfo.InvariantCulture, $"{Clock(segment.Start, vtt)} --> {Clock(segment.End, vtt)}\n{segment.Text.Trim()}\n\n");
        }

        return text.ToString();
    }

    private static string Clock(TimeSpan time, bool vtt)
        => string.Create(CultureInfo.InvariantCulture, $"{(int)time.TotalHours:00}:{time.Minutes:00}:{time.Seconds:00}{(vtt ? '.' : ',')}{time.Milliseconds:000}");

    private static string? NullIfEmpty(string? value) => string.IsNullOrWhiteSpace(value) ? null : value;
}
