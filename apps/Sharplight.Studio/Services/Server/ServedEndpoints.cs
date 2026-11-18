using Sharplight.Models;
using Sharplight.Studio.Models;

namespace Sharplight.Studio.Services.Server;

/// <summary>Which HTTP endpoints serve the models of each kind (what the embedded server maps over the runtime's host).</summary>
internal static class ServedEndpoints
{
    /// <summary>Chat and completion endpoints of the OpenAI API.</summary>
    public static IReadOnlyList<string> OpenAiText { get; } = ["/v1/chat/completions", "/v1/completions"];

    /// <summary>Messages endpoints of the Anthropic API.</summary>
    public static IReadOnlyList<string> AnthropicText { get; } = ["/v1/messages", "/v1/messages/count_tokens"];

    /// <summary>The paths a model of <paramref name="kind"/> answers on; empty for kinds the engine does not serve over HTTP.</summary>
    /// <param name="kind">The model's kind (<see cref="ModelKinds"/>, or <see cref="ModelKindCatalog.Embeddings"/>).</param>
    /// <param name="openAi">The OpenAI API is on.</param>
    /// <param name="anthropic">The Anthropic API is on.</param>
    public static IReadOnlyList<string> For(string kind, bool openAi, bool anthropic)
    {
        return kind switch
        {
            ModelKinds.TextGeneration => [.. openAi ? OpenAiText : [], .. anthropic ? AnthropicText : []],
            ModelKinds.SpeechRecognition => ["/v1/audio/transcriptions", "/v1/audio/translations"],
            ModelKinds.SpeechSynthesis => ["/v1/audio/speech"],
            ModelKinds.ImageGeneration => ["/v1/images/generations"],
            ModelKindCatalog.Embeddings => ["/v1/embeddings"],
            _ => [],
        };
    }
}
