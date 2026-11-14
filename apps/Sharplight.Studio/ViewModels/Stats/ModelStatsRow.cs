namespace Sharplight.Studio.ViewModels.Stats;

/// <summary>Aggregates of one model.</summary>
/// <param name="Model">Model id.</param>
/// <param name="Requests">Requests.</param>
/// <param name="Tokens">Generated tokens.</param>
/// <param name="Decode">Decode speed.</param>
/// <param name="Prefill">Prefill speed.</param>
/// <param name="Ttft">Median time to first token.</param>
internal sealed record ModelStatsRow(string Model, string Requests, string Tokens, string Decode, string Prefill, string Ttft);
