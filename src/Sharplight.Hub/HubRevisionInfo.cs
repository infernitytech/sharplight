namespace Sharplight.Hub;

/// <summary>An <c>api/models/{repo}/revision/{rev}</c> answer: the canonical id, the resolved commit and whether the repository is gated.</summary>
internal readonly record struct HubRevisionInfo(string? Id, string Commit, bool Gated);
