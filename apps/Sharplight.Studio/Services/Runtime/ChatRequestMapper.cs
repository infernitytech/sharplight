using System.Text;
using Sharplight.Llm;
using Sharplight.Llm.Serving;
using Sharplight.Llm.Serving.Chat;
using Sharplight.Studio.Models;

namespace Sharplight.Studio.Services.Runtime;

/// <summary>Turns the app's chat input into engine requests.</summary>
internal static class ChatRequestMapper
{
    /// <summary>Requests from the app are charged to this tenant, which tells them apart from API requests in the statistics.</summary>
    public const string StudioTenant = "sharplight-studio";

    /// <summary>The engine request for a text-only conversation.</summary>
    public static ChatRequest ToChatRequest(ChatInput input, bool reasoningTemplate)
    {
        return new ChatRequest
        {
            Messages = [.. input.Messages.Select(message => new ChatTurn(RoleName(message.Role), message.Text))],
            Sampling = ToSamplingParams(input.Sampling),
            MaxTokens = Math.Max(0, input.Sampling.MaxTokens),
            EnableThinking = reasoningTemplate ? input.EnableThinking : null,
            Tenant = StudioTenant,
        };
    }

    /// <summary>The engine's sampling parameters.</summary>
    public static SamplingParams ToSamplingParams(SamplingSettings sampling)
    {
        return new SamplingParams
        {
            Temperature = (float)sampling.Temperature,
            TopP = (float)sampling.TopP,
            TopK = sampling.TopK,
            MinP = (float)sampling.MinP,
            RepetitionPenalty = (float)sampling.RepetitionPenalty,
            Seed = sampling.Seed,
            Greedy = sampling.Temperature <= 0,
        };
    }

    /// <summary>The session sampler's options (vision path).</summary>
    public static SamplingOptions ToSamplingOptions(SamplingSettings sampling)
    {
        return new SamplingOptions
        {
            Temperature = (float)sampling.Temperature,
            TopP = (float)sampling.TopP,
            TopK = sampling.TopK,
            MinP = (float)sampling.MinP,
            RepetitionPenalty = (float)sampling.RepetitionPenalty,
            Seed = sampling.Seed,
        };
    }

    /// <summary>Statistics from the engine's usage and timings.</summary>
    public static MessageStats ToStats(GenerationUsage usage, GenerationTimings? timings, ChatFinishReason? finish)
    {
        var t = timings ?? default;
        return new MessageStats
        {
            PromptTokens = usage.PromptTokens,
            CachedTokens = usage.CachedTokens,
            CompletionTokens = usage.CompletionTokens,
            QueueMs = t.QueueWait.TotalMilliseconds,
            PrefillMs = t.Prefill.TotalMilliseconds,
            DecodeMs = t.Decode.TotalMilliseconds,
            TotalMs = t.Total.TotalMilliseconds,
            DraftTokens = t.DraftTokens,
            AcceptedDraftTokens = t.AcceptedDraftTokens,
            Finish = FinishName(finish),
        };
    }

    /// <summary>Statistics of a request the engine reported as finished.</summary>
    public static MessageStats ToStats(FinishedRequest finished)
    {
        var stats = ToStats(finished.Usage, finished.Timings, null);
        return stats with { Finish = finished.Finish.ToString().ToLowerInvariant() };
    }

    /// <summary>The lower-case finish reason (<c>stop</c>, <c>length</c>, ...).</summary>
    public static string FinishName(ChatFinishReason? finish)
    {
        return finish switch
        {
            null => "stop",
            ChatFinishReason.ToolCalls => "tool_calls",
            var reason => reason.Value.ToString().ToLowerInvariant(),
        };
    }

    /// <summary>The chat template role name.</summary>
    public static string RoleName(ChatRole role)
    {
        return role switch
        {
            ChatRole.System => "system",
            ChatRole.Assistant => "assistant",
            _ => "user",
        };
    }

    /// <summary>Tool call text as shown in the chat: <c>name(arguments)</c>.</summary>
    public static string FormatToolCall(string name, string arguments)
    {
        return new StringBuilder().Append(name).Append('(').Append(arguments).Append(')').ToString();
    }
}
