namespace Sharplight.Llm.Serving;

/// <summary>A request the engine finished, as reported to <see cref="EngineOptions.RequestFinished"/>.</summary>
/// <param name="Id">The request id (<see cref="GenerationHandle.Id"/>).</param>
/// <param name="Tenant">The request's <see cref="GenerationRequest.Tenant"/>.</param>
/// <param name="Finish">Why it ended.</param>
/// <param name="Usage">Token counts.</param>
/// <param name="Timings">Where the time went.</param>
/// <param name="Error">Failure message when <paramref name="Finish"/> is <see cref="FinishReason.Error"/>.</param>
public sealed record FinishedRequest(long Id, string? Tenant, FinishReason Finish, GenerationUsage Usage, GenerationTimings Timings, string? Error);
