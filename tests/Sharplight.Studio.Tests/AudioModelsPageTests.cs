using Sharplight.Audio;
using Sharplight.Models;
using Sharplight.Speech;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Library;
using Sharplight.Studio.Tests.Fakes;
using Sharplight.Studio.ViewModels.Audio;
using Sharplight.Studio.ViewModels.Inference;
using Sharplight.Studio.ViewModels.Inference.Parameters;
using Sharplight.Studio.ViewModels.Speech;

namespace Sharplight.Studio.Tests;

/// <summary>The Audio, Speak and Transcribe pages with the audio models of 2026-11-19 (ACE-Step, VibeVoice, CSM, OmniVoice, VibeVoice-ASR) over fake task clients.</summary>
public sealed class AudioModelsPageTests
{
    private static async Task WaitFor(Func<bool> condition)
    {
        for (int i = 0; i < 400 && !condition(); i++)
        {
            await Task.Delay(10, TestContext.Current.CancellationToken);
        }

        Assert.True(condition());
    }

    private static bool Visible(ParameterPanelViewModel panel, string key) => panel.Parameters.First(parameter => parameter.Key == key).IsVisibleNow;

    private static FakeModel Generator(TestServices services, string id, out FakeAudioGenerator client)
    {
        client = new FakeAudioGenerator();
        var model = new FakeModel(id) { Kind = ModelKinds.AudioGeneration };
        model.Clients[typeof(IAudioGenerator)] = client;
        services.Runtime.Add(model);
        return model;
    }

    private static FakeModel Synthesizer(TestServices services, string id, FakeSynthesizer client)
    {
        var model = new FakeModel(id) { Kind = ModelKinds.SpeechSynthesis };
        model.Clients[typeof(ISpeechSynthesizer)] = client;
        services.Runtime.Add(model);
        return model;
    }

    // ---- Audio page: diffusion music ----

    [Fact]
    public async Task Audio_DiffusionMusic_OffersLyricsLanguageStepsAndShift_AndHidesWhatTurboDoesNotUse()
    {
        using var services = new TestServices();
        Generator(services, "ace-step-1.5-turbo", out _);
        var page = new AudioPageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();

        Assert.True(page.IsMusic);
        Assert.True(page.IsLyricMusic);
        foreach (string key in new[] { "duration", "seed", "musicsteps", "shift", "musiclanguage" })
        {
            Assert.True(Visible(page.Panel, key), key);
        }

        foreach (string key in new[] { "guidance", "greedy", "temperature", "musictopk" })
        {
            Assert.False(Visible(page.Panel, key), key);
        }
    }

    [Fact]
    public async Task Audio_NonDistilledDiffusionModel_ShowsGuidance()
    {
        using var services = new TestServices();
        Generator(services, "ace-step-1.5-base", out _);
        var page = new AudioPageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();

        Assert.True(Visible(page.Panel, "guidance"));
        Assert.False(Visible(page.Panel, "greedy"));
    }

    [Fact]
    public async Task Audio_TokenMusic_HasNoLyricsButSamplingControls()
    {
        using var services = new TestServices();
        services.Runtime.Add(new FakeModel("musicgen-small") { Kind = ModelKinds.MusicGeneration });
        var page = new AudioPageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();

        Assert.True(page.IsMusic);
        Assert.False(page.IsLyricMusic);
        foreach (string key in new[] { "guidance", "greedy", "temperature", "musictopk" })
        {
            Assert.True(Visible(page.Panel, key), key);
        }

        foreach (string key in new[] { "musicsteps", "shift", "musiclanguage" })
        {
            Assert.False(Visible(page.Panel, key), key);
        }
    }

    [Fact]
    public async Task Audio_DiffusionMusic_SendsLyricsLanguageStepsShiftAndSeed_AndKeepsTheLyricsInHistory()
    {
        using var services = new TestServices();
        Generator(services, "ace-step-1.5-turbo", out var client);
        client.MaxDuration = TimeSpan.FromSeconds(20);
        var page = new AudioPageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();
        page.Prompt = "upbeat pop";
        page.Lyrics = "[verse]\nHello world";
        page.Panel.Apply(new Dictionary<string, string> { ["musiclanguage"] = "en", ["musicsteps"] = "12", ["shift"] = "2.5", ["seed"] = "7", ["duration"] = "45" });
        page.RunCommand.Execute(null);
        await page.Queue.Items.Single().Completion.WaitAsync(TestContext.Current.CancellationToken);

        var request = Assert.Single(client.Requests);
        Assert.Equal("upbeat pop", request.Prompt);
        Assert.Equal("[verse]\nHello world", request.Lyrics);
        Assert.Equal("en", request.Language);
        Assert.Equal(12, request.Steps);
        Assert.Equal(2.5f, request.Shift);
        Assert.Equal(7, request.Seed);
        Assert.Null(request.GuidanceScale);   // turbo takes none
        Assert.Equal(TimeSpan.FromSeconds(20), request.Duration);   // limited by the model
        Assert.Equal("[verse]\nHello world", page.History.Items.Single().Record.Input["lyrics"]);
    }

    [Fact]
    public async Task Audio_DiffusionMusic_DefaultsLeaveStepsShiftAndLanguageToTheModel()
    {
        using var services = new TestServices();
        Generator(services, "ace-step-1.5-turbo", out var client);
        var page = new AudioPageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();
        page.Prompt = "ambient";
        page.RunCommand.Execute(null);
        await page.Queue.Items.Single().Completion.WaitAsync(TestContext.Current.CancellationToken);

        var request = Assert.Single(client.Requests);
        Assert.Null(request.Steps);
        Assert.Null(request.Shift);
        Assert.Null(request.Language);
        Assert.Null(request.Lyrics);
    }

    // ---- Kind detection ----

    [Fact]
    public void Inspector_RecognizesTheAceStepFolderLayout_WithoutAModelType()
    {
        using var directory = new TempDir();
        string root = directory.File("ACE-Step-1.5");
        Directory.CreateDirectory(Path.Combine(root, "acestep-v15-turbo"));
        Directory.CreateDirectory(Path.Combine(root, "vae"));
        Directory.CreateDirectory(Path.Combine(root, "Qwen3-Embedding-0.6B"));
        File.WriteAllText(Path.Combine(root, "acestep-v15-turbo", "config.json"), """{"architectures":["AceStepConditionGenerationModel"]}""");

        var inspected = ModelInspector.Inspect(root);

        Assert.Equal(ModelKinds.AudioGeneration, inspected.Kind);
        Assert.True(inspected.KindInferred);
    }

    [Fact]
    public void Inspector_DoesNotTakeAFolderWithoutTheThreeParts_ForAceStep()
    {
        using var directory = new TempDir();
        string root = directory.File("partial");
        Directory.CreateDirectory(Path.Combine(root, "acestep-v15-turbo"));
        File.WriteAllText(Path.Combine(root, "acestep-v15-turbo", "config.json"), "{}");
        Directory.CreateDirectory(Path.Combine(root, "vae"));

        Assert.NotEqual(ModelKinds.AudioGeneration, ModelInspector.Inspect(root).Kind);
    }

    [Theory]
    [InlineData("VibeVoice-1.5B", """{"model_type":"vibevoice","architectures":["VibeVoiceForConditionalGeneration"]}""", ModelKinds.SpeechSynthesis)]
    [InlineData("VibeVoice-ASR", """{"model_type":"vibevoice","architectures":["VibeVoiceForASRTraining"]}""", ModelKinds.SpeechRecognition)]
    [InlineData("VibeVoice-ASR-HF", """{"model_type":"vibevoice_asr","architectures":["VibeVoiceAsrForConditionalGeneration"]}""", ModelKinds.TextGeneration)]
    [InlineData("csm-1b", """{"model_type":"csm","architectures":["CsmForConditionalGeneration"]}""", ModelKinds.SpeechSynthesis)]
    [InlineData("OmniVoice", """{"model_type":"omnivoice"}""", ModelKinds.SpeechSynthesis)]
    [InlineData("Dia-1.6B-0626", """{"model_type":"dia","architectures":["DiaForConditionalGeneration"]}""", ModelKinds.SpeechSynthesis)]
    [InlineData("Dia-1.6B", """{"data":{"channels":9,"delay_pattern":[0,8,9,10,11,12,13,14,15]},"model":{"decoder":{"n_layer":18},"encoder":{"n_layer":12}},"version":"0.1"}""", ModelKinds.SpeechSynthesis)]
    public void Inspector_RecognizesTheNewSpeechFolders_ByTheirConfiguration(string folder, string config, string kind)
    {
        using var directory = new TempDir();
        string path = directory.File(folder);
        Directory.CreateDirectory(path);
        File.WriteAllText(Path.Combine(path, "config.json"), config);

        Assert.Equal(kind, ModelInspector.Inspect(path).Kind);
    }

    // ---- Speak ----

    [Fact]
    public async Task Speak_VibeVoice_OffersTheScriptSpeakersAndDiffusionOptions()
    {
        using var services = new TestServices();
        var client = new FakeSynthesizer("vibevoice");
        Synthesizer(services, "vibevoice-1.5b", client);
        var page = new SpeakPageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();
        await WaitFor(() => page.IsMultiSpeaker);

        Assert.True(Visible(page.Panel, "diffsteps"));
        Assert.True(Visible(page.Panel, "guidance"));
        Assert.False(Visible(page.Panel, "language"));
        Assert.False(Visible(page.Panel, "design"));
        Assert.False(page.HasSpeakers);   // no labelled line yet

        page.InputText = "Speaker 1: Hello there.\nSpeaker 2: Hi!\nSpeaker 1: How are you?";
        Assert.Equal(["1", "2"], page.Speakers.Select(speaker => speaker.Speaker));
        Assert.True(page.HasSpeakers);

        page.InputText = "Speaker 2: Only me now.";
        Assert.Equal(["2"], page.Speakers.Select(speaker => speaker.Speaker));
    }

    [Fact]
    public async Task Speak_VibeVoice_SpeaksTheScriptInOneRequest_WithTheVoiceOfEachSpeaker()
    {
        using var services = new TestServices();
        var client = new FakeSynthesizer("vibevoice");
        Synthesizer(services, "vibevoice-1.5b", client);
        var page = new SpeakPageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();
        await WaitFor(() => page.IsMultiSpeaker);

        // A saved voice for speaker 1 and a reference recording for speaker 2.
        page.VoiceLibrary.Save(new VoiceProfile
        {
            Name = "Anna",
            Architecture = client.VoiceArchitecture,
            Tensors = new Dictionary<string, VoiceTensor> { ["speaker"] = VoiceTensor.From([1f, 2f, 3f]) },
        });
        await page.RefreshSavedVoicesAsync(null);
        string reference = Path.Combine(services.Directory, "bob.wav");
        WavFile.Write(reference, client.Tone(1));
        page.InputText = "Speaker 1: Hello there.\nSpeaker 2: Hi!";
        var first = page.Speakers[0];
        Assert.Contains(first.Choices, choice => choice.Value == "saved:Anna");
        first.Selected = first.Choices.First(choice => choice.Value == "saved:Anna");
        var second = page.Speakers[1];
        second.Selected = second.Choices.First(choice => choice.Value == SpeakerVoiceViewModel.ReferenceFile);
        second.ReferencePath = reference;
        second.ReferenceText = "some words";
        page.Panel.Apply(new Dictionary<string, string> { ["diffsteps"] = "20", ["guidance"] = "2" });

        page.RunCommand.Execute(null);
        await page.Queue.Items.Single().Completion.WaitAsync(TestContext.Current.CancellationToken);

        var (text, options, _) = Assert.Single(client.Requests);
        Assert.Equal("Speaker 1: Hello there.\nSpeaker 2: Hi!", text);
        Assert.Equal(20, options!.DiffusionSteps);
        Assert.Equal(2f, options.GuidanceScale);
        Assert.Equal("Anna", options.SpeakerVoices!["1"].Name);
        Assert.Equal("speaker-2", options.SpeakerVoices["2"].Name);
        Assert.Equal("some words", Assert.Single(client.Cloned).ReferenceText);
    }

    [Fact]
    public async Task Speak_VibeVoice_RefusesAReferenceSpeakerWithoutAFile()
    {
        using var services = new TestServices();
        var client = new FakeSynthesizer("vibevoice");
        Synthesizer(services, "vibevoice-1.5b", client);
        var page = new SpeakPageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();
        await WaitFor(() => page.IsMultiSpeaker);
        page.InputText = "Speaker 1: Hello.";
        page.Speakers[0].Selected = page.Speakers[0].Choices.First(choice => choice.Value == SpeakerVoiceViewModel.ReferenceFile);

        page.RunCommand.Execute(null);
        var item = page.Queue.Items.Single();
        await item.Completion.WaitAsync(TestContext.Current.CancellationToken);

        Assert.Equal(RunStatus.Failed, item.Status);

        Assert.Contains("Speaker 1 has no reference recording", page.Notice, StringComparison.Ordinal);
        Assert.Empty(client.Requests);
    }

    [Fact]
    public async Task Speak_Dia_ExplainsTheSpeakerTags_WithoutPerSpeakerVoices()
    {
        using var services = new TestServices();
        var client = new FakeSynthesizer("dia");
        Synthesizer(services, "dia-1.6b", client);
        var page = new SpeakPageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();
        await WaitFor(() => page.HasScriptHint);

        Assert.False(page.IsMultiSpeaker);
        Assert.Contains("[S1]", page.ScriptHint, StringComparison.Ordinal);
        Assert.Contains("(laughs)", page.ScriptHint, StringComparison.Ordinal);
    }

    [Fact]
    public async Task Speak_Csm_SpeaksTheTurnsOneByOne_WithTheSpeakerNumbersAsVoices()
    {
        using var services = new TestServices();
        var client = new FakeSynthesizer("csm") { Voices = ["0", "1"] };
        Synthesizer(services, "csm-1b", client);
        var page = new SpeakPageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();
        await WaitFor(() => page.IsMultiSpeaker);

        Assert.False(Visible(page.Panel, "diffsteps"));
        Assert.False(Visible(page.Panel, "voice"));   // the voices are chosen per speaker
        page.AddTurnCommand.Execute(null);
        Assert.Equal("Speaker 0: ", page.InputText);
        page.InputText += "Hello.";
        page.AddTurnCommand.Execute(null);
        Assert.EndsWith("\nSpeaker 1: ", page.InputText, StringComparison.Ordinal);
        page.InputText += "Hi, you.";
        Assert.Equal(["0", "1"], page.Speakers.Select(speaker => speaker.Speaker));
        Assert.Contains(page.Speakers[0].Choices, choice => choice.Value == "voice:1");
        Assert.DoesNotContain(page.Speakers[0].Choices, choice => choice.Value == SpeakerVoiceViewModel.ReferenceFile);

        // Speaker 0 speaks with the model's voice 1 instead.
        page.Speakers[0].Selected = page.Speakers[0].Choices.First(choice => choice.Value == "voice:1");
        page.RunCommand.Execute(null);
        await page.Queue.Items.Single().Completion.WaitAsync(TestContext.Current.CancellationToken);

        Assert.Equal(["Hello.", "Hi, you."], client.Requests.Select(request => request.Text));
        Assert.Equal(["1", "1"], client.Requests.Select(request => request.Options!.Voice));
        Assert.Contains("audio", page.Gallery.Items.Single().Facts.Select(fact => fact.Split(':')[0]));
    }

    [Fact]
    public async Task Speak_OmniVoice_OffersTheLanguagesAndAVoiceDesignBox()
    {
        using var services = new TestServices();
        var client = new FakeSynthesizer("omnivoice") { LanguageNames = ["abkhazian", "english", "german"] };
        Synthesizer(services, "omnivoice", client);
        var page = new SpeakPageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();
        await WaitFor(() => Visible(page.Panel, "language"));

        Assert.True(Visible(page.Panel, "design"));
        Assert.False(Visible(page.Panel, "instructions"));
        Assert.False(Visible(page.Panel, "diffsteps"));
        Assert.False(page.IsMultiSpeaker);
        var language = page.Panel.Parameters.OfType<ChoiceParameterViewModel>().First(parameter => parameter.Key == "language");
        Assert.Equal(["Automatic", "Abkhazian", "English", "German"], language.Choices.Select(choice => choice.Label));

        language.Deserialize("german");
        page.Panel.Parameters.OfType<TextParameterViewModel>().First(parameter => parameter.Key == "design").Deserialize("female, british accent");
        page.InputText = "Guten Tag.";
        page.RunCommand.Execute(null);
        await page.Queue.Items.Single().Completion.WaitAsync(TestContext.Current.CancellationToken);

        var (_, options, _) = Assert.Single(client.Requests);
        Assert.Equal("german", options!.Language);
        Assert.Equal("female, british accent", options.Instructions);
        Assert.Null(options.DiffusionSteps);
    }

    [Fact]
    public async Task Speak_PlainModel_HasNoScriptLanguageDiffusionOrDesignControls()
    {
        using var services = new TestServices();
        Synthesizer(services, "outetts", new FakeSynthesizer());
        var page = new SpeakPageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();

        Assert.False(page.IsMultiSpeaker);
        Assert.Equal("", page.ScriptHint);
        foreach (string key in new[] { "diffsteps", "guidance", "language", "design" })
        {
            Assert.False(Visible(page.Panel, key), key);
        }

        Assert.True(Visible(page.Panel, "instructions"));
        page.InputText = "plain text";
        page.RunCommand.Execute(null);
        await page.Queue.Items.Single().Completion.WaitAsync(TestContext.Current.CancellationToken);
        var (_, options, _) = Assert.Single(((FakeSynthesizer)(await ((FakeModel)services.Runtime.LoadedModels.Single()).AcquireClientAsync<ISpeechSynthesizer>(CancellationToken.None)).Client).Requests);
        Assert.Null(options!.Language);
        Assert.Null(options.GuidanceScale);
        Assert.Null(options.SpeakerVoices);
    }

    [Fact]
    public void SpeechScript_ParsesTurnsAndSpeakers()
    {
        var turns = SpeechScript.Parse("Speaker 1: Hello\nand more\nSpeaker 2: Hi\n");
        Assert.Equal([new SpeechScript.Turn("1", "Hello and more"), new SpeechScript.Turn("2", "Hi")], turns);
        Assert.Equal([new SpeechScript.Turn("0", "no labels")], SpeechScript.Parse("no labels"));
        Assert.Empty(SpeechScript.Speakers("no labels"));
        Assert.Equal(["a", "b"], SpeechScript.Speakers("speaker a: x\nSpeaker b: y\nSpeaker a: z"));
    }

    // ---- Transcribe: speakers ----

    private static List<TranscriptionSegment> TwoSpeakers() =>
    [
        new() { Start = TimeSpan.Zero, End = TimeSpan.FromSeconds(2), Text = " Hello, how are you?", Speaker = "0" },
        new() { Start = TimeSpan.FromSeconds(2), End = TimeSpan.FromSeconds(4), Text = " Fine, thanks.", Speaker = "1" },
        new() { Start = TimeSpan.FromSeconds(4), End = TimeSpan.FromSeconds(5), Text = " And you?", Speaker = "1" },
        new() { Start = TimeSpan.FromSeconds(5), End = TimeSpan.FromSeconds(7), Text = " Great.", Speaker = "0" },
    ];

    [Fact]
    public async Task Transcribe_ShowsTheSpeakersOfEverySegment_WithOneColourEach_AndExportsThem()
    {
        using var services = new TestServices();
        var client = new FakeRecognizer();
        client.Script.Clear();
        client.Script.AddRange(TwoSpeakers());
        var model = new FakeModel("vibevoice-asr") { Kind = ModelKinds.SpeechRecognition };
        model.Clients[typeof(ISpeechRecognizer)] = client;
        services.Runtime.Add(model);
        var page = new TranscribePageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();
        string wav = services.Directory + "/talk.wav";
        WavFile.Write(wav, new AudioBuffer(new float[16000 * 8], 16000));
        await page.SetAudioAsync(wav);
        page.RunCommand.Execute(null);
        await page.Queue.Items.Single().Completion.WaitAsync(TestContext.Current.CancellationToken);
        await WaitFor(() => page.Segments.Count == 4);

        Assert.True(page.HasSpeakers);
        Assert.Equal(["Speaker 0", "Speaker 1"], page.SpeakerLabels);
        Assert.Equal("2 speakers: Speaker 0, Speaker 1", page.SpeakerSummary);
        Assert.Equal(["Speaker 0", "Speaker 1", "Speaker 1", "Speaker 0"], page.Segments.Select(segment => segment.Speaker));
        Assert.Equal(page.Segments[0].Color, page.Segments[3].Color);
        Assert.NotEqual(page.Segments[0].Color, page.Segments[1].Color);

        string target = services.Directory + "/out.txt";
        services.Platform.SaveTo = target;
        await page.ExportCommand.ExecuteAsync("txt");
        Assert.Equal("Speaker 0: Hello, how are you?\nSpeaker 1: Fine, thanks. And you?\nSpeaker 0: Great.\n", await File.ReadAllTextAsync(target, TestContext.Current.CancellationToken));
        await page.ExportCommand.ExecuteAsync("srt");
        Assert.Contains("00:00:02,000 --> 00:00:04,000\nSpeaker 1: Fine, thanks.", await File.ReadAllTextAsync(target, TestContext.Current.CancellationToken), StringComparison.Ordinal);
        await page.ExportCommand.ExecuteAsync("vtt");
        Assert.Contains("00:00:00.000 --> 00:00:02.000\n<v Speaker 0>Hello, how are you?", await File.ReadAllTextAsync(target, TestContext.Current.CancellationToken), StringComparison.Ordinal);

        // The stored run keeps the speakers: showing it again restores them.
        page.Segments.Clear();
        page.History.RestoreCommand.Execute(page.History.Items.Single());
        Assert.Equal(["Speaker 0", "Speaker 1"], page.SpeakerLabels);
        Assert.Equal(4, page.Segments.Count);
    }

    [Fact]
    public async Task Transcribe_WithoutSpeakers_KeepsThePlainExports()
    {
        using var services = new TestServices();
        var model = new FakeModel("whisper-tiny") { Kind = ModelKinds.SpeechRecognition };
        model.Clients[typeof(ISpeechRecognizer)] = new FakeRecognizer();
        services.Runtime.Add(model);
        var page = new TranscribePageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();
        string wav = services.Directory + "/a.wav";
        WavFile.Write(wav, new AudioBuffer(new float[16000 * 7], 16000));
        await page.SetAudioAsync(wav);
        page.RunCommand.Execute(null);
        await page.Queue.Items.Single().Completion.WaitAsync(TestContext.Current.CancellationToken);

        Assert.False(page.HasSpeakers);
        Assert.Equal("", page.SpeakerSummary);
        string target = services.Directory + "/plain.vtt";
        services.Platform.SaveTo = target;
        await page.ExportCommand.ExecuteAsync("vtt");
        string vtt = await File.ReadAllTextAsync(target, TestContext.Current.CancellationToken);
        Assert.DoesNotContain("<v ", vtt, StringComparison.Ordinal);
    }

    [Fact]
    public void TranscriptFile_LabelsSpeakersOnlyWhenTheModelGaveThem()
    {
        var file = new TranscriptFile
        {
            Segments = [new(0, 1, "Hi", "Speaker A"), new(1, 2, "there", null), new(2, 3, "again", "Speaker A")],
        };

        Assert.Equal(["Speaker A"], file.Speakers);
        Assert.Equal("Speaker A: Hi\nthere\nSpeaker A: again", file.ToText());
        Assert.Equal("Speaker 3", new TranscriptSegmentRecord(0, 1, "x", "3").SpeakerLabel);
        Assert.Equal("", new TranscriptSegmentRecord(0, 1, "x").SpeakerLabel);
    }

    // ---- Licence notices ----

    [Theory]
    [InlineData("other", true)]
    [InlineData("OTHER", true)]
    [InlineData("cc-by-nc-4.0", true)]
    [InlineData("flux-dev-non-commercial-license", true)]
    [InlineData("apache-2.0", false)]
    [InlineData("mit", false)]
    [InlineData("cc-by-4.0", false)]
    [InlineData("openrail++", true)]
    [InlineData("gemma", true)]
    [InlineData("llama3.3", true)]
    [InlineData("cdla-permissive-2.0", false)]
    [InlineData(null, false)]
    public void LicenseNotices_AreRequiredForEveryNonPermissiveLicence(string? license, bool required) => Assert.Equal(required, LicenseNotices.Requires(license));

    [Fact]
    public void LicenseNotices_ShowOnlyTheLicenseNotice_NotTheGeneralNotes_OrAGenericLine()
    {
        static Sharplight.Hub.Catalog.CatalogEntry Entry(string? license, string? notes, string? notice = null) =>
            new() { Id = "m", Name = "M", Kind = ModelKinds.SpeechSynthesis, License = license, Notes = notes, LicenseNotice = notice };

        Assert.Equal("Weights are CC-BY-NC.", LicenseNotices.For(Entry("other", "hybrid Mamba-2 / attention", "Weights are CC-BY-NC.")));
        var generic = LicenseNotices.For(Entry("other", "hybrid Mamba-2 / attention"));
        Assert.Equal($"Licence: other. {LicenseNotices.GenericLine}", generic);
        Assert.DoesNotContain("Mamba", generic, StringComparison.Ordinal);
        Assert.Null(LicenseNotices.For(Entry("mit", "fast")));
        Assert.Null(LicenseNotices.For(Entry(null, "fast")));
    }

    [Fact]
    public async Task Speak_ShowsTheLicenceNoticeOfAModelInstalledFromTheCatalog()
    {
        using var services = new TestServices();
        var client = new FakeSynthesizer("omnivoice") { LanguageNames = ["english"] };
        var model = Synthesizer(services, "omnivoice", client);
        services.Library.Entries.Add(new LibraryEntry
        {
            Id = "omnivoice",
            DisplayName = "k2-fsa/OmniVoice",
            Path = model.Path,
            Kind = ModelKinds.SpeechSynthesis,
            Manifest = new Sharplight.Hub.ModelManifest { Repository = "k2-fsa/OmniVoice", Tag = "latest", Alias = "omnivoice", Commit = "abc", Catalog = "omnivoice/safetensors" },
        });
        services.Hub.Catalog = new Sharplight.Hub.Catalog.VerifiedCatalog
        {
            Updated = "2026-11-19",
            Entries = [new Sharplight.Hub.Catalog.CatalogEntry { Id = "omnivoice", Name = "OmniVoice", Kind = ModelKinds.SpeechSynthesis, License = "other", Notes = "unrelated remark", LicenseNotice = "weights are CC-BY-NC (non-commercial)" }],
        };
        var page = new SpeakPageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();
        await WaitFor(() => page.HasLicenseNotice);

        Assert.Equal("weights are CC-BY-NC (non-commercial)", page.LicenseNotice);

        // Dismissed once: it is remembered in the settings and does not come back when the model is chosen again.
        page.License.DismissCommand.Execute(null);
        Assert.False(page.HasLicenseNotice);
        Assert.Equal(["omnivoice"], services.Settings.Current.DismissedLicenseNotices);
        await page.License.RefreshAsync(model);
        Assert.False(page.HasLicenseNotice);
    }
}
