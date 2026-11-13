using Sharplight.Samples;

namespace Sharplight.Serve;

/// <summary>Everything the command line selects, before it is merged with the optional configuration file.</summary>
/// <param name="PositionalModel">Model given as the positional argument, or null.</param>
/// <param name="PositionalModelName">Id of the positional model (default: file or directory name).</param>
/// <param name="ModelArguments">Values of repeated <c>--model NAME=PATH[,key=value...]</c> options.</param>
/// <param name="ConfigPath">JSON configuration file, or null.</param>
/// <param name="Runtime">Device and sizing defaults for models that do not override them.</param>
/// <param name="ListenUrl">Address Kestrel listens on.</param>
/// <param name="Api">Exposed API dialects.</param>
/// <param name="ApiKeys">Accepted API keys; empty means unauthenticated.</param>
/// <param name="CorsOrigins">Allowed CORS origins.</param>
/// <param name="Slots">Default concurrent sequences per model.</param>
/// <param name="MaxQueue">Waiting requests per model before answering 429.</param>
/// <param name="PrefillChunk">Prompt rows per step and sequence.</param>
/// <param name="StepTokens">Token budget per engine step; null uses the model's batch size.</param>
/// <param name="NgramDraftTokens">Draft length of prompt-lookup speculation; 0 disables it.</param>
/// <param name="MmprojPath">Default vision projector.</param>
/// <param name="ChatTemplatePath">Chat template file replacing the models' own.</param>
/// <param name="DefaultTemperature">Temperature for requests that give none.</param>
/// <param name="VramReserveBytes">Device memory kept free; null defers to the config file, then the built-in default.</param>
/// <param name="VramBudgetBytes">Total device memory cap; null defers to the config file.</param>
/// <param name="MaxLoadedModels">Resident model limit; null defers to the config file.</param>
/// <param name="Lazy">Load models on first request.</param>
/// <param name="IdleTimeoutSeconds">Idle unload timeout; null defers to the config file.</param>
internal sealed record ServeSettings(
    string? PositionalModel,
    string? PositionalModelName,
    IReadOnlyList<string> ModelArguments,
    string? ConfigPath,
    RuntimeSettings Runtime,
    string ListenUrl,
    ApiSurface Api,
    IReadOnlyList<string> ApiKeys,
    IReadOnlyList<string> CorsOrigins,
    int Slots,
    int MaxQueue,
    int PrefillChunk,
    int? StepTokens,
    int NgramDraftTokens,
    string? MmprojPath,
    string? ChatTemplatePath,
    float DefaultTemperature,
    long? VramReserveBytes,
    long? VramBudgetBytes,
    int? MaxLoadedModels,
    bool Lazy,
    double? IdleTimeoutSeconds);
