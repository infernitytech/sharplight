namespace Sharplight.Studio.Models;

/// <summary>A conversation as listed in the sidebar (without its messages).</summary>
/// <param name="Id">The conversation id.</param>
/// <param name="Title">Its title.</param>
/// <param name="UpdatedAt">Last change.</param>
/// <param name="MessageCount">Messages in all branches.</param>
internal sealed record ConversationSummary(string Id, string Title, DateTimeOffset UpdatedAt, int MessageCount);
