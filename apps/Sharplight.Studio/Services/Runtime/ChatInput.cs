using Sharplight.Studio.Models;

namespace Sharplight.Studio.Services.Runtime;

/// <summary>A chat request from the app: the conversation and how to sample.</summary>
/// <param name="Messages">System, user and assistant messages, oldest first.</param>
/// <param name="Sampling">Sampling settings and the answer length limit.</param>
/// <param name="EnableThinking">Let reasoning models think first (passed to the chat template).</param>
internal sealed record ChatInput(IReadOnlyList<ChatInputMessage> Messages, SamplingSettings Sampling, bool EnableThinking)
{
    /// <summary>True when any message carries an image.</summary>
    public bool HasImages => Messages.Any(message => message.Images.Count > 0);
}
