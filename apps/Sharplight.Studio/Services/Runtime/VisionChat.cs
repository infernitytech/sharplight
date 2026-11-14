using System.Diagnostics;
using Microsoft.Extensions.AI;
using Sharplight.Extensions.AI;
using Sharplight.Llm;
using Sharplight.Studio.Models;
using AiChatMessage = Microsoft.Extensions.AI.ChatMessage;
using AiChatRole = Microsoft.Extensions.AI.ChatRole;
using StudioChatRole = Sharplight.Studio.Models.ChatRole;

namespace Sharplight.Studio.Services.Runtime;

/// <summary>
/// Image conversations on a model with a vision tower: a <see cref="SessionGenerationBackend"/> (one sequence, created with the
/// model's multimodal decode graph and vision encoder on first use) behind a <see cref="SharplightChatClient"/>, which renders
/// the chat template with image placeholders and parses reasoning. Requests run one at a time.
/// </summary>
internal sealed class VisionChat : IDisposable
{
    private readonly SessionGenerationBackend _backend;

    public VisionChat(LlmModel model)
    {
        _backend = new SessionGenerationBackend(model, ownsModel: false);
    }

    /// <summary>Generates the answer, reporting content and reasoning through <paramref name="emit"/>; returns its statistics.</summary>
    public async Task<MessageStats> RunAsync(ChatInput input, bool reasoningTemplate, Action<ChatStreamUpdate> emit, CancellationToken cancellationToken)
    {
        var options = new SharplightChatClientOptions
        {
            Sampling = ChatRequestMapper.ToSamplingOptions(input.Sampling),
            DefaultMaxOutputTokens = input.Sampling.MaxTokens > 0 ? input.Sampling.MaxTokens : 4096,
            EnableThinking = reasoningTemplate && input.EnableThinking,
        };
        using var client = new SharplightChatClient(_backend, options, ownsBackend: false);
        var messages = input.Messages.Select(ToAiMessage).ToList();
        var stopwatch = Stopwatch.StartNew();
        string finish = "stop";
        await foreach (var update in client.GetStreamingResponseAsync(messages, null, cancellationToken).ConfigureAwait(false))
        {
            foreach (var content in update.Contents)
            {
                switch (content)
                {
                    case TextReasoningContent reasoning when reasoning.Text.Length > 0:
                        emit(ChatStreamUpdate.Reasoning(reasoning.Text));
                        break;
                    case TextContent text when text.Text.Length > 0:
                        emit(ChatStreamUpdate.Content(text.Text));
                        break;
                    case FunctionCallContent call:
                        emit(new ChatStreamUpdate(ChatStreamKind.ToolCall, ChatRequestMapper.FormatToolCall(call.Name, string.Join(", ", call.Arguments?.Select(pair => $"{pair.Key}={pair.Value}") ?? []))));
                        break;
                }
            }

            if (update.FinishReason is { } reason)
            {
                finish = reason == ChatFinishReason.Length ? "length" : reason == ChatFinishReason.ToolCalls ? "tool_calls" : "stop";
            }
        }

        var total = stopwatch.Elapsed;
        var last = _backend.LastStats;
        return new MessageStats
        {
            PromptTokens = last?.PromptTokens ?? 0,
            CompletionTokens = last?.GeneratedTokens ?? 0,
            PrefillMs = last?.PrefillTime.TotalMilliseconds ?? 0,
            DecodeMs = last?.DecodeTime.TotalMilliseconds ?? 0,
            TotalMs = total.TotalMilliseconds,
            Finish = finish,
        };
    }

    /// <summary>Disposes the session and the vision encoder (not the model).</summary>
    public void Dispose() => _backend.Dispose();

    private static AiChatMessage ToAiMessage(ChatInputMessage message)
    {
        var role = message.Role switch
        {
            StudioChatRole.System => AiChatRole.System,
            StudioChatRole.Assistant => AiChatRole.Assistant,
            _ => AiChatRole.User,
        };
        var contents = new List<AIContent>();
        contents.AddRange(message.Images.Select(image => new DataContent(image.Data, image.MediaType)));
        contents.Add(new TextContent(message.Text));
        return new AiChatMessage(role, contents);
    }
}
