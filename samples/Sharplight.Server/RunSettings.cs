using Sharplight.Samples.Hub;
using Sharplight.Samples;

namespace Sharplight.Server;

/// <summary>Parsed <c>run</c> options.</summary>
/// <param name="Model">The model as typed: a file, a stored model or a Hub repository.</param>
/// <param name="Runtime">Device and sizing.</param>
/// <param name="Pull">How a missing model is pulled.</param>
/// <param name="Temperature">Sampling temperature (0 = greedy).</param>
/// <param name="MaxTokens">Maximum tokens per answer.</param>
/// <param name="SystemPrompt">System message that starts the conversation, or null.</param>
/// <param name="Prompt">Answer this one prompt and exit instead of chatting, or null.</param>
/// <param name="Think">Let reasoning models think before answering (templates that support switching it).</param>
internal sealed record RunSettings(
    string Model,
    RuntimeSettings Runtime,
    PullSettings Pull,
    float Temperature,
    int MaxTokens,
    string? SystemPrompt,
    string? Prompt,
    bool Think);
