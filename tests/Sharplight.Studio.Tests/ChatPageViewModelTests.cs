using Sharplight.Studio.Tests.Fakes;
using Sharplight.Studio.ViewModels;

namespace Sharplight.Studio.Tests;

public sealed class ChatPageViewModelTests
{
    private static async Task WaitFor(Func<bool> condition)
    {
        for (int i = 0; i < 300 && !condition(); i++)
        {
            await Task.Delay(10, TestContext.Current.CancellationToken);
        }

        Assert.True(condition());
    }

    [Fact]
    public async Task Activate_OpensTheNewestConversation_WithItsModel()
    {
        using var services = new TestServices();
        await DemoData.FillAsync(services);
        var main = new MainWindowViewModel(services.Services);
        var chat = main.Chat;

        await chat.ActivateAsync();
        await WaitFor(() => chat.Current is not null);

        Assert.Equal(3, chat.Conversations.Count);
        Assert.Equal("Calling the server from C#", chat.Conversations[0].Title);
        Assert.Equal(2, chat.Current!.Messages.Count);
        Assert.Equal("qwen3-0.6b", chat.SelectedModel?.Id);
    }

    [Fact]
    public async Task TwoConversations_GenerateConcurrently_OnDifferentModels()
    {
        using var services = new TestServices();
        var first = services.Runtime.Add(new FakeModel("first") { Gate = new TaskCompletionSource() });
        var second = services.Runtime.Add(new FakeModel("second"));
        var main = new MainWindowViewModel(services.Services);
        var chat = main.Chat;
        await chat.ActivateAsync();
        await WaitFor(() => chat.Current is not null);

        chat.SelectedModel = chat.Models.Single(model => model.Id == "first");
        var one = chat.Current!;
        one.InputText = "slow question";
        var sending = one.SendCommand.ExecuteAsync(null);
        await WaitFor(() => one.IsGenerating);

        await chat.NewChatCommand.ExecuteAsync(null);
        var two = chat.Current!;
        Assert.NotSame(one, two);
        chat.SelectedModel = chat.Models.Single(model => model.Id == "second");
        two.InputText = "fast question";
        await two.SendCommand.ExecuteAsync(null);

        Assert.Equal("Hello there!", two.Messages[1].Content);
        Assert.True(one.IsGenerating);   // still streaming in the background
        Assert.Contains(chat.Conversations, item => item.Id == one.Id && item.IsGenerating);

        first.Gate!.SetResult();
        await sending;
        Assert.Equal("Hello there!", one.Messages[1].Content);
        Assert.Equal(("first", "second"), (one.ModelId, two.ModelId));
    }

    [Fact]
    public async Task DeletingTheLastConversation_StartsANewOne()
    {
        using var services = new TestServices();
        var chat = new MainWindowViewModel(services.Services).Chat;
        await chat.ActivateAsync();
        await WaitFor(() => chat.Current is not null);
        var only = Assert.Single(chat.Conversations);

        await chat.DeleteConversationCommand.ExecuteAsync(only);

        var fresh = Assert.Single(chat.Conversations);
        Assert.NotEqual(only.Id, fresh.Id);
        Assert.Equal(fresh.Id, chat.Current?.Id);
        Assert.Equal("New chat", fresh.Title);
    }
}
