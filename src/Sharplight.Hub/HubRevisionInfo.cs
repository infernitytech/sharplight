namespace Sharplight.Hub;

/// <summary>
/// An <c>api/models/{repo}/revision/{rev}</c> answer: the canonical id, the resolved commit, whether the repository is gated and
/// the model family the Hub reports (GGUF <c>general.architecture</c> or the checkpoint's <c>config.json</c> <c>model_type</c>).
/// </summary>
internal readonly record struct HubRevisionInfo(string? Id, string Commit, bool Gated, string? Architecture = null);
