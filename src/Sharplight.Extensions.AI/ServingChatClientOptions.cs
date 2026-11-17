using Sharplight.Llm.Serving;

namespace Sharplight.Extensions.AI;

/// <summary>
/// Settings of a <see cref="ServingChatClient"/>: who the requests are charged to, which prefix-cache partition they use and the
/// defaults for options a call leaves out. Per call, <see cref="Microsoft.Extensions.AI.ChatOptions.AdditionalProperties"/> may override
/// the tenant, partition and priority (<see cref="ServingChatClient.TenantProperty"/>, <see cref="ServingChatClient.CachePartitionProperty"/>,
/// <see cref="ServingChatClient.PriorityProperty"/>).
/// </summary>
public sealed record ServingChatClientOptions
{
    /// <summary>Model id reported in responses (default: the service's model name).</summary>
    public string? ModelId { get; init; }

    /// <summary>Fair-share tenant of the requests (<see cref="GenerationRequest.Tenant"/>), or null.</summary>
    public string? Tenant { get; init; }

    /// <summary>Prefix-cache partition of the requests (<see cref="GenerationRequest.CachePartition"/>), e.g. a tenant's isolation scope.</summary>
    public string? CachePartition { get; init; }

    /// <summary>Scheduling priority (higher runs first).</summary>
    public int Priority { get; init; }

    /// <summary>Sampling parameters a call leaves out.</summary>
    public SamplingParams DefaultSampling { get; init; } = new();

    /// <summary>Generated tokens when a call names no limit (0 = up to the context).</summary>
    public int DefaultMaxTokens { get; init; }

    /// <summary>System prompt put first when the messages have no system message, or null.</summary>
    public string? DefaultSystemPrompt { get; init; }

    /// <summary>Total time budget per request, or null.</summary>
    public TimeSpan? Timeout { get; init; }
}
