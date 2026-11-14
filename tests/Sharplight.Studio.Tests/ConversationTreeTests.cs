using Sharplight.Studio.Models;

namespace Sharplight.Studio.Tests;

public sealed class ConversationTreeTests
{
    [Fact]
    public void Append_MovesTheLeaf_AndActivePathFollowsTheParents()
    {
        var conversation = new Conversation();
        var user = new ChatMessageRecord { Role = ChatRole.User, Content = "hi" };
        conversation = ConversationTree.Append(conversation, user);
        var answer = new ChatMessageRecord { ParentId = user.Id, Role = ChatRole.Assistant, Content = "hello" };
        conversation = ConversationTree.Append(conversation, answer);

        Assert.Equal(answer.Id, conversation.CurrentLeafId);
        Assert.Equal([user.Id, answer.Id], ConversationTree.ActivePath(conversation).Select(message => message.Id));
    }

    [Fact]
    public void SwitchVersion_MovesBetweenSiblings_AndLandsOnTheirNewestLeaf()
    {
        var start = DateTimeOffset.UtcNow;
        var user = new ChatMessageRecord { Role = ChatRole.User, Content = "q", CreatedAt = start };
        var first = new ChatMessageRecord { ParentId = user.Id, Role = ChatRole.Assistant, Content = "a1", CreatedAt = start.AddSeconds(1) };
        var followUp = new ChatMessageRecord { ParentId = first.Id, Role = ChatRole.User, Content = "more", CreatedAt = start.AddSeconds(2) };
        var second = new ChatMessageRecord { ParentId = user.Id, Role = ChatRole.Assistant, Content = "a2", CreatedAt = start.AddSeconds(3) };
        var conversation = new Conversation { Messages = [user, first, followUp, second], CurrentLeafId = second.Id };

        Assert.Equal([first.Id, second.Id], ConversationTree.Siblings(conversation, second).Select(message => message.Id));

        var back = ConversationTree.SwitchVersion(conversation, second, -1);
        Assert.Equal(followUp.Id, back.CurrentLeafId);   // the branch's newest leaf, not the sibling itself

        var forward = ConversationTree.SwitchVersion(back, first, 1);
        Assert.Equal(second.Id, forward.CurrentLeafId);
        Assert.Same(forward, ConversationTree.SwitchVersion(forward, second, 1));   // already the last version
    }

    [Fact]
    public void Replace_KeepsTheTreeAndUpdatesTheMessage()
    {
        var message = new ChatMessageRecord { Role = ChatRole.Assistant };
        var conversation = ConversationTree.Append(new Conversation(), message);
        var replaced = ConversationTree.Replace(conversation, message with { Content = "done" });
        Assert.Equal("done", Assert.Single(replaced.Messages).Content);
        Assert.Equal(message.Id, replaced.CurrentLeafId);
    }

    [Fact]
    public void PathTo_StopsOnCycles()
    {
        var a = new ChatMessageRecord { Id = "a", ParentId = "b" };
        var b = new ChatMessageRecord { Id = "b", ParentId = "a" };
        Assert.Equal(2, ConversationTree.PathTo(new Conversation { Messages = [a, b] }, "a").Count);
    }

    [Theory]
    [InlineData("  Hello world \nsecond line", "Hello world")]
    [InlineData("", "New chat")]
    [InlineData("A very long first line that certainly does not fit into the title of the conversation", "A very long first line that certainly does not…")]
    public void TitleFrom_UsesTheFirstLine(string text, string expected)
    {
        Assert.Equal(expected, ConversationTree.TitleFrom(text));
    }
}
