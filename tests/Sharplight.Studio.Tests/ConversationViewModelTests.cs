using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.Tests.Fakes;
using Sharplight.Studio.ViewModels.Chat;

namespace Sharplight.Studio.Tests;

public sealed class ConversationViewModelTests
{
    private static ConversationViewModel Create(TestServices services, Func<ILoadedModel?> model, List<Conversation>? changes = null) =>
        new(new Conversation { SystemPrompt = "Be nice." }, services.Services, model, conversation => changes?.Add(conversation));

    [Fact]
    public async Task Send_StreamsTheAnswer_StoresStats_AndTitlesTheChat()
    {
        using var services = new TestServices();
        var model = services.Runtime.Add(new FakeModel("qwen3", ModelCapabilities.Chat | ModelCapabilities.Reasoning));
        var changes = new List<Conversation>();
        var chat = Create(services, () => model, changes);

        chat.InputText = "Hello model\nsecond line";
        await chat.SendCommand.ExecuteAsync(null);

        Assert.Equal(2, chat.Messages.Count);
        var answer = chat.Messages[1];
        Assert.Equal(("Hello there!", "Thinking about it.", false), (answer.Content, answer.Reasoning, answer.IsStreaming));
        Assert.Equal(3, answer.Stats?.CompletionTokens);
        Assert.Equal("Hello model", chat.Conversation.Title);
        Assert.Equal("qwen3", chat.Conversation.ModelId);
        Assert.Equal("", chat.InputText);
        Assert.False(chat.IsGenerating);

        var input = Assert.Single(model.Inputs);
        Assert.Equal([ChatRole.System, ChatRole.User], input.Messages.Select(message => message.Role));
        Assert.Equal("Be nice.", input.Messages[0].Text);

        var stored = await services.Conversations.LoadAsync(chat.Id);
        Assert.Equal(2, stored!.Messages.Count);
        Assert.NotEmpty(changes);
    }

    [Fact]
    public async Task Send_WithoutAModel_ExplainsWhatToDo()
    {
        using var services = new TestServices();
        var chat = Create(services, () => null);
        chat.InputText = "hi";
        await chat.SendCommand.ExecuteAsync(null);
        Assert.Empty(chat.Messages);
        Assert.Contains("Load a model", chat.Notice);
    }

    [Fact]
    public async Task Images_AreRefused_ForModelsWithoutVision()
    {
        using var services = new TestServices();
        var model = services.Runtime.Add(new FakeModel("text-only"));
        var chat = Create(services, () => model);
        chat.AddImage([0x89, 0x50, 0x4E, 0x47], "image/png", "x.png");
        chat.InputText = "What is this?";
        Assert.True(chat.ImageNotSupported);
        await chat.SendCommand.ExecuteAsync(null);
        Assert.Empty(model.Inputs);
        Assert.Contains("cannot read images", chat.Notice);
    }

    [Fact]
    public async Task Regenerate_AddsAVersion_AndTheSwitcherMovesBetweenThem()
    {
        using var services = new TestServices();
        var model = services.Runtime.Add(new FakeModel("m"));
        var chat = Create(services, () => model);
        chat.InputText = "q";
        await chat.SendCommand.ExecuteAsync(null);
        model.Script[1] = ChatStreamUpdate.Content("Second");

        await chat.RegenerateCommand.ExecuteAsync(chat.Messages[1]);

        var answer = chat.Messages[1];
        Assert.Equal(2, chat.Messages.Count);
        Assert.Equal(("Second there!", 2, 2), (answer.Content, answer.VersionIndex, answer.VersionCount));
        await chat.PreviousVersionCommand.ExecuteAsync(answer);
        Assert.Equal(("Hello there!", 1), (chat.Messages[1].Content, chat.Messages[1].VersionIndex));
    }

    [Fact]
    public async Task EditingAQuestion_BranchesTheConversation()
    {
        using var services = new TestServices();
        var model = services.Runtime.Add(new FakeModel("m"));
        var chat = Create(services, () => model);
        chat.InputText = "first";
        await chat.SendCommand.ExecuteAsync(null);

        var question = chat.Messages[0];
        chat.EditCommand.Execute(question);
        question.EditText = "edited";
        await chat.SaveEditCommand.ExecuteAsync(question);

        Assert.Equal(("edited", 2, 2), (chat.Messages[0].Content, chat.Messages[0].VersionIndex, chat.Messages[0].VersionCount));
        Assert.Equal(4, chat.Conversation.Messages.Count);
        Assert.Equal("edited", model.Inputs[^1].Messages[^1].Text);
    }

    [Fact]
    public async Task Stop_CancelsTheStream_AndKeepsThePartialAnswer()
    {
        using var services = new TestServices();
        var model = services.Runtime.Add(new FakeModel("m") { Gate = new TaskCompletionSource() });
        var chat = Create(services, () => model);
        chat.InputText = "q";
        var sending = chat.SendCommand.ExecuteAsync(null);
        for (int i = 0; i < 200 && !(chat.IsGenerating && chat.Messages.Count == 2 && chat.Messages[1].Reasoning.Length > 0); i++)
        {
            await Task.Delay(5, TestContext.Current.CancellationToken);
        }

        Assert.True(chat.IsGenerating);
        chat.StopCommand.Execute(null);
        await sending;

        Assert.False(chat.IsGenerating);
        Assert.Equal("Thinking about it.", chat.Messages[1].Reasoning);
        Assert.Equal("", chat.Messages[1].Content);
    }

    [Fact]
    public async Task SwitchingTheModel_AppliesToTheNextAnswer()
    {
        using var services = new TestServices();
        var first = services.Runtime.Add(new FakeModel("first"));
        var second = services.Runtime.Add(new FakeModel("second"));
        ILoadedModel current = first;
        var chat = Create(services, () => current);
        chat.InputText = "one";
        await chat.SendCommand.ExecuteAsync(null);
        current = second;
        chat.ModelId = "second";
        chat.InputText = "two";
        await chat.SendCommand.ExecuteAsync(null);

        Assert.Equal(["first", "second"], chat.Messages.Where(message => message.IsAssistant).Select(message => message.ModelName));
        Assert.Equal(3, second.Inputs[0].Messages.Count(message => message.Role != ChatRole.System));
        Assert.Equal("second", (await services.Conversations.LoadAsync(chat.Id))!.ModelId);
    }
}
