using Sharplight.Audio;
using Sharplight.ImageGeneration;
using Sharplight.Models;
using Sharplight.Speech;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.Tests.Fakes;
using Sharplight.Studio.ViewModels.Chat;

namespace Sharplight.Studio.Tests;

/// <summary>Chat with voice (dictation, reading aloud) and pictures (<c>/image</c>) over fake task clients.</summary>
public sealed class ChatMediaTests
{
    private static ConversationViewModel Create(TestServices services, ChatMediaViewModel media, Func<ILoadedModel?> model) =>
        new(new Conversation(), services.Services, model, _ => { }, media);

    private static FakeModel Chat(TestServices services) => services.Runtime.Add(new FakeModel("qwen3"));

    private static FakeModel Recognizer(TestServices services, string id = "whisper-tiny")
    {
        var model = new FakeModel(id) { Kind = ModelKinds.SpeechRecognition };
        model.Clients[typeof(ISpeechRecognizer)] = new FakeRecognizer();
        return services.Runtime.Add(model);
    }

    private static FakeModel Synthesizer(TestServices services, out FakeSynthesizer client, string id = "outetts")
    {
        client = new FakeSynthesizer();
        var model = new FakeModel(id) { Kind = ModelKinds.SpeechSynthesis };
        model.Clients[typeof(ISpeechSynthesizer)] = client;
        return services.Runtime.Add(model);
    }

    private static FakeModel Painter(TestServices services, out FakeImageGenerator client, string id = "sdxl-turbo")
    {
        client = new FakeImageGenerator();
        var model = new FakeModel(id) { Kind = ModelKinds.ImageGeneration };
        model.Clients[typeof(IImageGenerator)] = client;
        return services.Runtime.Add(model);
    }

    private static void RecordsAWav(TestServices services)
        => services.Audio.OnRecorded = path => WavFile.Write(path, new AudioBuffer(new float[16000], 16000));

    [Fact]
    public async Task Dictation_RecordsTheMicrophone_AndPutsTheTranscriptIntoThePrompt()
    {
        using var services = new TestServices();
        var chatModel = Chat(services);
        Recognizer(services);
        RecordsAWav(services);
        var media = new ChatMediaViewModel(services.Services);
        var chat = Create(services, media, () => chatModel);
        Assert.True(media.CanDictate);

        chat.InputText = "Please";
        await chat.DictateCommand.ExecuteAsync(null);
        Assert.True(media.IsRecording);
        await chat.DictateCommand.ExecuteAsync(null);

        Assert.False(media.IsRecording);
        Assert.False(media.IsTranscribing);
        Assert.Equal("Please And so, my fellow Americans, ask not what your country can do for you.", chat.InputText);
        Assert.Empty(chatModel.Inputs);   // dictation fills the box, the user sends
    }

    [Fact]
    public async Task Dictation_WithoutAMicrophoneOrAModel_SaysWhy()
    {
        using var services = new TestServices();
        var chat = Create(services, new ChatMediaViewModel(services.Services), () => null);
        Assert.False(chat.Media!.CanDictate);
        await chat.DictateCommand.ExecuteAsync(null);
        Assert.Contains("speech-to-text", chat.Notice, StringComparison.Ordinal);

        Recognizer(services);
        services.Audio.IsAvailable = false;
        await chat.DictateCommand.ExecuteAsync(null);
        Assert.Equal("No audio device.", chat.Notice);
    }

    [Fact]
    public async Task ReadAloud_SpeaksFinishedAnswers_WithoutMarkup_AndStreamsThePlayback()
    {
        using var services = new TestServices();
        var chatModel = Chat(services);
        chatModel.Script.Clear();
        chatModel.Script.Add(ChatStreamUpdate.Content("**Hello** there, see [docs](http://x.y).\n\n```cs\nvar x = 1;\n```\n\n- done"));
        chatModel.Script.Add(ChatStreamUpdate.Finished(new MessageStats { PromptTokens = 1, CompletionTokens = 5 }));
        Synthesizer(services, out var tts);
        var media = new ChatMediaViewModel(services.Services) { ReadAloud = true };
        Assert.True(services.Settings.Current.Voice.ReadAloud);
        var chat = Create(services, media, () => chatModel);

        chat.InputText = "hi";
        await chat.SendCommand.ExecuteAsync(null);
        await WaitFor(() => services.Audio.Streamed.Count > 0 && !media.IsSpeaking);

        var request = Assert.Single(tts.Requests);
        Assert.True(request.Streaming);
        Assert.Equal("Hello there, see docs. done", request.Text);
        Assert.Equal(5, services.Audio.Streamed.Count);
    }

    [Fact]
    public async Task ReadAloudButton_SpeaksAMessage_AndStopsWhileSpeaking()
    {
        using var services = new TestServices();
        var chatModel = Chat(services);
        Synthesizer(services, out var tts);
        var media = new ChatMediaViewModel(services.Services);
        var chat = Create(services, media, () => chatModel);
        chat.InputText = "hi";
        await chat.SendCommand.ExecuteAsync(null);
        Assert.Empty(tts.Requests);   // off by default

        tts.StepWork = TimeSpan.FromMilliseconds(20);
        var speaking = chat.ReadAloudCommand.ExecuteAsync(chat.Messages[1]);
        await WaitFor(() => media.IsSpeaking);
        await chat.ReadAloudCommand.ExecuteAsync(chat.Messages[1]);   // a second click stops
        await speaking;
        Assert.False(media.IsSpeaking);
        Assert.True(services.Audio.Streamed.Count < 3);
    }

    [Fact]
    public async Task ReadAloud_OffersTheVoicesOfTheModel_UsesTheChosenOne_AndRemembersIt()
    {
        using var services = new TestServices();
        var chatModel = Chat(services);
        Synthesizer(services, out var tts);
        tts.Voices = ["af_heart", "bm_george"];
        var media = new ChatMediaViewModel(services.Services);
        await media.RefreshVoicesAsync();
        Assert.True(media.HasVoices);
        Assert.Equal([ChatMediaViewModel.DefaultVoice, "af_heart", "bm_george"], media.Voices);
        Assert.Equal(ChatMediaViewModel.DefaultVoice, media.SelectedVoice);
        var chat = Create(services, media, () => chatModel);
        chat.InputText = "hi";
        await chat.SendCommand.ExecuteAsync(null);

        await chat.ReadAloudCommand.ExecuteAsync(chat.Messages[1]);
        Assert.Null(tts.Requests[0].Options);   // the model's own voice

        media.SelectedVoice = "bm_george";
        Assert.Equal("bm_george", services.Settings.Current.Voice.SynthesisVoice);
        await chat.ReadAloudCommand.ExecuteAsync(chat.Messages[1]);
        Assert.Equal("bm_george", tts.Requests[1].Options!.Voice);

        // A new chat window starts with the remembered voice; a model without that voice falls back without forgetting it.
        var again = new ChatMediaViewModel(services.Services);
        await again.RefreshVoicesAsync();
        Assert.Equal("bm_george", again.SelectedVoice);
        tts.Voices = ["other"];
        await again.RefreshVoicesAsync();
        Assert.Equal(ChatMediaViewModel.DefaultVoice, again.SelectedVoice);
        Assert.Equal("bm_george", services.Settings.Current.Voice.SynthesisVoice);

        // A model that offers no voices shows no choice.
        tts.Voices = [];
        await again.RefreshVoicesAsync();
        Assert.False(again.HasVoices);
    }

    [Fact]
    public async Task ImageCommand_StoresThePictureInTheConversation_AndKeepsItOutOfTheChatModel()
    {
        using var services = new TestServices();
        var chatModel = Chat(services);
        Painter(services, out var generator);
        var media = new ChatMediaViewModel(services.Services);
        var chat = Create(services, media, () => chatModel);

        chat.InputText = "/image a lighthouse at dusk";
        await chat.SendCommand.ExecuteAsync(null);

        Assert.Empty(chatModel.Inputs);   // not a chat request
        Assert.Equal("a lighthouse at dusk", Assert.Single(generator.Requests).Request.Prompt);
        Assert.Equal(2, chat.Messages.Count);
        var picture = chat.Messages[1];
        var attachment = Assert.Single(picture.Attachments);
        Assert.True(attachment.IsImage);
        Assert.Equal(AttachmentKind.GeneratedImage, attachment.Record.Kind);
        Assert.Equal(0x89, attachment.ImageData![0]);   // PNG
        Assert.Contains("sdxl-turbo", picture.Content, StringComparison.Ordinal);
        Assert.Equal("a lighthouse at dusk", chat.Conversation.Title);

        // Stored with the conversation, picture included.
        var stored = await services.Conversations.LoadAsync(chat.Id);
        Assert.Equal(AttachmentKind.GeneratedImage, Assert.Single(stored!.Messages.Single(message => message.Role == ChatRole.Assistant).Attachments).Kind);
        Assert.True(stored.Messages.Single(message => message.Role == ChatRole.User).ImageRequest);

        // A following chat message does not carry the picture or its request to the language model.
        chat.InputText = "Thanks, now tell me a joke";
        await chat.SendCommand.ExecuteAsync(null);
        var input = Assert.Single(chatModel.Inputs);
        Assert.DoesNotContain(input.Messages, message => message.Text.Contains("lighthouse", StringComparison.Ordinal) || message.Images.Count > 0);
        Assert.Equal("Thanks, now tell me a joke", input.Messages.Last().Text);
    }

    [Fact]
    public async Task ImageCommand_WithoutAnImageModel_OrAPrompt_ExplainsWhatToDo()
    {
        using var services = new TestServices();
        var chatModel = Chat(services);
        var media = new ChatMediaViewModel(services.Services);
        var chat = Create(services, media, () => chatModel);
        chat.InputText = "/image a cat";
        await chat.SendCommand.ExecuteAsync(null);
        Assert.Contains("image-generation model", chat.Notice, StringComparison.Ordinal);
        Assert.Empty(chat.Messages);

        Painter(services, out _);
        chat.InputText = "/image";
        await chat.SendCommand.ExecuteAsync(null);
        Assert.Contains("Describe the picture", chat.Notice, StringComparison.Ordinal);
        Assert.Empty(chat.Messages);

        Assert.False(ConversationViewModel.TryParseImageCommand("/imagine something", out _));
        Assert.True(ConversationViewModel.TryParseImageCommand("/IMAGE  a cat ", out var prompt));
        Assert.Equal("a cat", prompt);
    }

    [Fact]
    public async Task RegeneratingAPicture_AddsAVersion_AndAFailureIsShownOnTheMessage()
    {
        using var services = new TestServices();
        var chatModel = Chat(services);
        var painter = Painter(services, out _);
        var media = new ChatMediaViewModel(services.Services);
        var chat = Create(services, media, () => chatModel);
        chat.InputText = "/image a cat";
        await chat.SendCommand.ExecuteAsync(null);

        await chat.RegenerateCommand.ExecuteAsync(chat.Messages[1]);
        Assert.Equal(2, chat.Messages[1].VersionCount);
        Assert.Empty(chatModel.Inputs);

        painter.Clients.Clear();
        await chat.RegenerateCommand.ExecuteAsync(chat.Messages[1]);
        Assert.Equal(3, chat.Messages[1].VersionCount);
        Assert.True(chat.Messages[1].HasError);
        Assert.Empty(chat.Messages[1].Attachments);
    }

    [Fact]
    public void TheChosenModels_AreRemembered_AndRestoredWhenTheyAreResidentAgain()
    {
        using var services = new TestServices();
        Recognizer(services, "whisper-a");
        Recognizer(services, "whisper-b");
        var media = new ChatMediaViewModel(services.Services);
        Assert.Equal("whisper-a", media.Recognizers.SelectedModel!.Id);
        media.Recognizers.SelectedModel = media.Recognizers.Models[1];
        Assert.Equal("whisper-b", services.Settings.Current.Voice.RecognitionModel);

        var again = new ChatMediaViewModel(services.Services);
        Assert.Equal("whisper-b", again.Recognizers.SelectedModel!.Id);

        // A remembered model that is not resident yet is chosen when it appears, and the fallback is not remembered.
        services.Settings.Update(settings => settings with { Voice = settings.Voice with { SynthesisModel = "kokoro" } });
        var third = new ChatMediaViewModel(services.Services);
        Synthesizer(services, out _, "outetts");
        Assert.Equal("outetts", third.Synthesizers.SelectedModel!.Id);
        Assert.Equal("kokoro", services.Settings.Current.Voice.SynthesisModel);
        Synthesizer(services, out _, "kokoro");
        Assert.Equal("outetts", third.Synthesizers.SelectedModel!.Id);   // an existing choice is not taken away
    }

    [Theory]
    [InlineData("# Title\n\nSome *emphasis* and `code`.", "Title Some emphasis and code.")]
    [InlineData("1. one\n2. two", "one two")]
    [InlineData("| a | b |\n|---|---|\n| 1 | 2 |", ", a , b , , 1 , 2 ,")]
    [InlineData("```py\nprint(1)\n```", "")]
    public void SpeakableText_DropsMarkup(string markdown, string spoken) => Assert.Equal(spoken, SpeakableText.From(markdown));

    private static async Task WaitFor(Func<bool> condition)
    {
        for (int i = 0; i < 500 && !condition(); i++)
        {
            await Task.Delay(10, TestContext.Current.CancellationToken);
        }

        Assert.True(condition());
    }
}
