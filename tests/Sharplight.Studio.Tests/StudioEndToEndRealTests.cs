using System.Text.Json;
using Sharplight.Audio;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.Services.Server;
using Sharplight.Studio.ViewModels.Chat;
using Sharplight.Studio.ViewModels.Search;

namespace Sharplight.Studio.Tests;

/// <summary>
/// The new Studio features end to end on real small models (skipped when a model is not on disk): voice chat (dictation, answer, reading
/// aloud), pictures in the chat, the Search page with embeddings and a reranker, and the embedded API server answering every kind.
/// Everything runs headless through the view models and the engine runtime on the GPU.
/// </summary>
[Collection(LargeModelCollection.Name)]
public sealed class StudioEndToEndRealTests
{
    private static readonly string[] s_files =
    [
        "cats.txt|Cats are small domesticated carnivores. A cat sleeps most of the day, purrs when it is content and chases mice at night.",
        "ships.txt|The container ship left the harbour at dawn. The captain steered the vessel through the strait while the engine room crew checked the diesel engines.",
        "tax.txt|Income tax returns are due in April. You can deduct mortgage interest and charitable donations from your taxable income.",
        "pasta.txt|To cook spaghetti, boil salted water, add the pasta for ten minutes and drain it. Serve with tomato sauce and fresh basil.",
        "space.txt|The Voyager probes left the solar system. Jupiter is the largest planet and has a great red spot, a storm larger than Earth.",
    ];

    private static float[] Jfk() => WavFile.Read(RealStudioSession.Require("audio/jfk.wav")).Samples.ToArray();

    private static async Task WaitFor(Func<bool> condition, int seconds = 120)
    {
        for (int i = 0; i < seconds * 20 && !condition(); i++)
        {
            await Task.Delay(50, TestContext.Current.CancellationToken);
        }

        Assert.True(condition());
    }

    private static ConversationViewModel Chat(RealStudioSession session, ILoadedModel model, ChatMediaViewModel media)
    {
        var chat = new ConversationViewModel(new Conversation(), session.Services, () => model, _ => { }, media)
        {
            EnableThinking = false,
            Temperature = 0,
            MaxTokens = 48,
        };
        return chat;
    }

    private static bool IsAudible(IEnumerable<float> samples) => samples.Any(sample => MathF.Abs(sample) > 0.02f);

    [Fact]
    public async Task VoiceChat_DictatesTheMicrophone_AnswersWithQwen3_AndReadsTheAnswerAloudWithKokoro()
    {
        string whisper = RealStudioSession.Require("whisper.cpp/ggml-tiny.en.bin");
        string qwen = RealStudioSession.Require("gguf/Qwen3-0.6B/Qwen3-0.6B-Q8_0.gguf");
        string kokoro = RealStudioSession.Require("tts/kokoro");
        var jfk = Jfk();
        using var session = new RealStudioSession();
        var ct = TestContext.Current.CancellationToken;
        var chatModel = await session.LoadAsync("qwen3", qwen, ModelKinds.TextGeneration, ct);
        await session.LoadAsync("whisper-tiny", whisper, ModelKinds.SpeechRecognition, ct);
        await session.LoadAsync("kokoro", kokoro, ModelKinds.SpeechSynthesis, ct);
        session.Audio.Microphone = jfk;

        var media = new ChatMediaViewModel(session.Services) { ReadAloud = true };
        var chat = Chat(session, chatModel, media);
        Assert.True(media.CanDictate);
        Assert.True(media.CanSpeak);
        await WaitFor(() => media.HasVoices);
        Assert.Contains("af_heart", media.Voices);

        // The microphone path: record, stop, transcribe off the UI thread.
        await chat.DictateCommand.ExecuteAsync(null);
        Assert.True(media.IsRecording);
        await chat.DictateCommand.ExecuteAsync(null);
        Assert.Contains("ask not what your country can do for you", chat.InputText, StringComparison.OrdinalIgnoreCase);

        // The chat answer from the real language model, read aloud by itself once it finished.
        await chat.SendCommand.ExecuteAsync(null);
        Assert.Equal(2, chat.Messages.Count);
        var answer = chat.Messages[1];
        Assert.False(answer.HasError, answer.Content);
        Assert.NotEmpty(answer.Content.Trim());
        await WaitFor(() => session.Audio.Played.Count > 0);
        await WaitFor(() => !media.IsSpeaking);
        Assert.Equal(24000, session.Audio.PlayedSampleRate);
        Assert.True(session.Audio.Played.Count > 24000, $"{session.Audio.Played.Count} samples of speech");
        Assert.True(IsAudible(session.Audio.Played));
        Assert.True(string.IsNullOrEmpty(chat.Notice), chat.Notice);
    }

    [Fact]
    public async Task ReadAloud_UsesTheVoiceChosenInTheSettings_OfTheQwen3TtsCustomVoiceModel()
    {
        string tts = RealStudioSession.Require("tts/Qwen3-TTS-12Hz-0.6B-CustomVoice");
        string qwen = RealStudioSession.Require("gguf/Qwen3-0.6B/Qwen3-0.6B-Q8_0.gguf");
        using var session = new RealStudioSession();
        var ct = TestContext.Current.CancellationToken;
        var chatModel = await session.LoadAsync("qwen3", qwen, ModelKinds.TextGeneration, ct);
        await session.LoadAsync("qwen3-tts", tts, ModelKinds.SpeechSynthesis, ct);

        var media = new ChatMediaViewModel(session.Services);
        await WaitFor(() => media.HasVoices);
        Assert.Equal(ChatMediaViewModel.DefaultVoice, media.Voices[0]);
        Assert.True(media.Voices.Count > 2, string.Join(", ", media.Voices));
        string voice = media.Voices[^1];
        media.SelectedVoice = voice;
        Assert.Equal(voice, session.Services.Settings.Current.Voice.SynthesisVoice);

        var chat = Chat(session, chatModel, media);
        chat.InputText = "Say hello in one short sentence.";
        await chat.SendCommand.ExecuteAsync(null);
        await chat.ReadAloudCommand.ExecuteAsync(chat.Messages[1]);

        Assert.True(string.IsNullOrEmpty(chat.Notice), chat.Notice);
        Assert.Equal(24000, session.Audio.PlayedSampleRate);
        Assert.True(session.Audio.Played.Count > 12000);
        Assert.True(IsAudible(session.Audio.Played));
    }

    [Fact]
    public async Task ImageCommand_GeneratesWithARealDiffusersModel_StoresThePng_AndKeepsItOutOfTheNextTurn()
    {
        string sd = RealStudioSession.Require("sd/tiny-random-sd/diffusers");
        string qwen = RealStudioSession.Require("gguf/Qwen3-0.6B/Qwen3-0.6B-Q8_0.gguf");
        using var session = new RealStudioSession();
        var ct = TestContext.Current.CancellationToken;
        var chatModel = await session.LoadAsync("qwen3", qwen, ModelKinds.TextGeneration, ct);
        await session.LoadAsync("tiny-sd", sd, ModelKinds.ImageGeneration, ct);
        var media = new ChatMediaViewModel(session.Services);
        var chat = Chat(session, chatModel, media);
        Assert.True(media.CanPaint);

        chat.InputText = "/image a lighthouse at dusk";
        await chat.SendCommand.ExecuteAsync(null);

        Assert.Equal(2, chat.Messages.Count);
        Assert.False(chat.Messages[1].HasError, chat.Messages[1].Content);
        var attachment = Assert.Single(chat.Messages[1].Attachments);
        Assert.Equal(AttachmentKind.GeneratedImage, attachment.Record.Kind);
        var decoded = PngCodec.Decode(attachment.ImageData!);
        Assert.True(decoded.Width >= 64 && decoded.Height >= 64);
        var stored = await session.Services.Conversations.LoadAsync(chat.Id);
        Assert.Equal(AttachmentKind.GeneratedImage, Assert.Single(stored!.Messages.Single(message => message.Role == ChatRole.Assistant).Attachments).Kind);

        chat.InputText = "Reply with the single word: ready";
        await chat.SendCommand.ExecuteAsync(null);
        Assert.Equal(4, chat.Messages.Count);
        Assert.False(chat.Messages[3].HasError, chat.Messages[3].Content);
        var input = chat.BuildInput(ConversationTree.ActivePath(chat.Conversation));
        Assert.DoesNotContain(input.Messages, message => message.Text.Contains("lighthouse", StringComparison.Ordinal) || message.Images.Count > 0);
        Assert.Equal(2, input.Messages.Count);   // the question and the answer of the second turn; the picture turn is not sent
    }

    [Fact]
    public async Task SearchPage_RanksAFolderOfTextFiles_WithARealEmbeddingModel_AndRerankingKeepsTheRightFile()
    {
        string embed = RealStudioSession.Require("hf/all-MiniLM-L6-v2");
        string rerank = RealStudioSession.Require("hf/ms-marco-MiniLM-L6-v2");
        using var session = new RealStudioSession();
        var ct = TestContext.Current.CancellationToken;
        await session.LoadAsync("minilm", embed, ModelKindCatalog.Embeddings, ct);
        await session.LoadAsync("ms-marco", rerank, ModelKindCatalog.Reranker, ct);

        string folder = Path.Combine(session.Directory, "docs");
        Directory.CreateDirectory(folder);
        foreach (string entry in s_files)
        {
            string[] parts = entry.Split('|');
            await File.WriteAllTextAsync(Path.Combine(folder, parts[0]), parts[1], ct);
        }

        var page = new SearchPageViewModel(session.Services, new Fakes.FakeNavigator());
        await page.ActivateAsync();
        Assert.True(page.CanRerank);
        page.SetFolder(folder);
        page.IndexCommand.Execute(null);
        await page.Queue.Items.Single().Completion.WaitAsync(TimeSpan.FromMinutes(3), ct);
        Assert.True(page.HasIndex);
        Assert.Equal("5 documents, 5 passages, minilm", page.IndexSummary);

        (string Query, string Expected)[] questions =
        [
            ("What pet purrs and chases mice?", "cats.txt"),
            ("How do I boil pasta?", "pasta.txt"),
            ("When do I have to file my income tax?", "tax.txt"),
            ("Which planet is the biggest?", "space.txt"),
            ("Who steers the vessel through the strait?", "ships.txt"),
        ];
        foreach (var (query, expected) in questions)
        {
            page.Query = query;
            page.SearchCommand.Execute(null);
            await page.Queue.Items.Last().Completion.WaitAsync(TimeSpan.FromMinutes(1), ct);
            Assert.Equal(expected, page.Results[0].FileName);
            Assert.True(page.Results[0].Hit.Score > page.Results[1].Hit.Score);
        }

        // Reranked: the cross-encoder reads the question with each passage; the right document stays first, scores are its own.
        page.Panel.Apply(new Dictionary<string, string> { ["rerank"] = "true", ["topk"] = "3" });
        foreach (var (query, expected) in questions)
        {
            page.Query = query;
            page.SearchCommand.Execute(null);
            await page.Queue.Items.Last().Completion.WaitAsync(TimeSpan.FromMinutes(1), ct);
            Assert.Equal(3, page.Results.Count);
            Assert.Equal(expected, page.Results[0].FileName);
            Assert.True(page.Results[0].Hit.Score >= page.Results[1].Hit.Score);
        }

        Assert.Contains("reranker: ms-marco", page.ResultSummary + string.Join(" ", page.History.Items[0].Record.Metadata.Select(pair => $"{pair.Key}: {pair.Value}")), StringComparison.Ordinal);
    }

    [Fact]
    public async Task EmbeddedServer_AnswersEveryKindOfRealModel_OverHttp()
    {
        string whisper = RealStudioSession.Require("whisper.cpp/ggml-tiny.en.bin");
        string kokoro = RealStudioSession.Require("tts/kokoro");
        string sd = RealStudioSession.Require("sd/tiny-random-sd/diffusers");
        string embed = RealStudioSession.Require("hf/all-MiniLM-L6-v2");
        string rerank = RealStudioSession.Require("hf/ms-marco-MiniLM-L6-v2");
        string jfk = RealStudioSession.Require("audio/jfk.wav");
        using var session = new RealStudioSession();
        var ct = TestContext.Current.CancellationToken;
        await session.LoadAsync("whisper", whisper, ModelKinds.SpeechRecognition, ct);
        await session.LoadAsync("kokoro", kokoro, ModelKinds.SpeechSynthesis, ct);
        await session.LoadAsync("sd", sd, ModelKinds.ImageGeneration, ct);
        await session.LoadAsync("minilm", embed, ModelKindCatalog.Embeddings, ct);
        await session.LoadAsync("ms-marco", rerank, ModelKindCatalog.Reranker, ct);

        await using var server = new KestrelApiServer(session.Runtime.Host);
        int port = 38000 + Random.Shared.Next(2000);
        await server.StartAsync(new ServerSettings { Host = "127.0.0.1", Port = port }, ct);
        using var http = new HttpClient { BaseAddress = new Uri(server.BaseUrl!), Timeout = TimeSpan.FromMinutes(5) };

        var models = await http.GetFromJsonAsync<JsonElement>("/v1/models", ct);
        var ids = models.GetProperty("data").EnumerateArray().Select(model => model.GetProperty("id").GetString()).Order().ToList();
        Assert.Equal(["kokoro", "minilm", "ms-marco", "sd", "whisper"], ids);

        using var form = new MultipartFormDataContent { { new ByteArrayContent(await File.ReadAllBytesAsync(jfk, ct)), "file", "jfk.wav" }, { new StringContent("whisper"), "model" } };
        using var transcription = await http.PostAsync("/v1/audio/transcriptions", form, ct);
        Assert.True(transcription.IsSuccessStatusCode, await transcription.Content.ReadAsStringAsync(ct));
        var transcript = await transcription.Content.ReadFromJsonAsync<JsonElement>(ct);
        Assert.Contains("ask not what your country can do for you", transcript.GetProperty("text").GetString(), StringComparison.OrdinalIgnoreCase);

        using var speech = await http.PostAsJsonAsync("/v1/audio/speech", new { model = "kokoro", input = "Hello world. This is Sharplight speaking.", voice = "af_heart" }, ct);
        Assert.True(speech.IsSuccessStatusCode, await speech.Content.ReadAsStringAsync(ct));
        var spoken = WavFile.Read(new MemoryStream(await speech.Content.ReadAsByteArrayAsync(ct)));
        Assert.True(spoken.Duration.TotalSeconds > 1.5, $"{spoken.Duration.TotalSeconds:0.0} s");
        Assert.True(IsAudible(spoken.Samples.ToArray()));

        using var image = await http.PostAsJsonAsync("/v1/images/generations", new { model = "sd", prompt = "a lighthouse", size = "128x64", n = 1 }, ct);
        Assert.True(image.IsSuccessStatusCode, await image.Content.ReadAsStringAsync(ct));
        var picture = await image.Content.ReadFromJsonAsync<JsonElement>(ct);
        var png = PngCodec.Decode(Convert.FromBase64String(picture.GetProperty("data")[0].GetProperty("b64_json").GetString()!));
        Assert.Equal((128, 64), (png.Width, png.Height));

        using var embeddings = await http.PostAsJsonAsync("/v1/embeddings", new { model = "minilm", input = new[] { "a cat purrs", "a ship sails", "a cat purrs" } }, ct);
        Assert.True(embeddings.IsSuccessStatusCode, await embeddings.Content.ReadAsStringAsync(ct));
        var data = (await embeddings.Content.ReadFromJsonAsync<JsonElement>(ct)).GetProperty("data");
        Assert.Equal(3, data.GetArrayLength());
        float[] first = [.. data[0].GetProperty("embedding").EnumerateArray().Select(value => value.GetSingle())];
        float[] third = [.. data[2].GetProperty("embedding").EnumerateArray().Select(value => value.GetSingle())];
        float[] other = [.. data[1].GetProperty("embedding").EnumerateArray().Select(value => value.GetSingle())];
        Assert.Equal(384, first.Length);
        Assert.True(Cosine(first, third) > 0.999f);
        Assert.True(Cosine(first, other) < 0.9f);

        using var ranked = await http.PostAsJsonAsync("/v1/rerank", new { model = "ms-marco", query = "How many people live in Berlin?", documents = new[] { "Berlin has a population of 3.5 million people.", "Pasta is boiled in salted water." } }, ct);
        Assert.True(ranked.IsSuccessStatusCode, await ranked.Content.ReadAsStringAsync(ct));
        var results = (await ranked.Content.ReadFromJsonAsync<JsonElement>(ct)).GetProperty("results");
        Assert.Equal(0, results[0].GetProperty("index").GetInt32());
        Assert.True(results[0].GetProperty("relevance_score").GetDouble() > results[1].GetProperty("relevance_score").GetDouble());
    }

    private static float Cosine(float[] a, float[] b)
    {
        float dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }

        return dot / MathF.Sqrt(na * nb);
    }
}
