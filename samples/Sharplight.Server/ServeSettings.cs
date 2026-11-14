using Sharplight.Samples.Hub;
using Sharplight.Samples;

namespace Sharplight.Server;

/// <summary>Everything the command line selects, before it is merged with the optional configuration file.</summary>
/// <param name="PositionalModels">Models given as positional arguments (files, stored models or Hub repositories).</param>
/// <param name="PositionalModelName">Id of the single positional model (default: file name or the name as given).</param>
/// <param name="ModelArguments">Values of repeated <c>--model NAME=MODEL[,key=value...]</c> options.</param>
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
/// <param name="KvCacheTokens">Paged KV cache size in tokens shared by the slots; null reserves the full context per slot.</param>
/// <param name="NgramDraftTokens">Draft length of prompt-lookup speculation; 0 disables it.</param>
/// <param name="MtpDraftTokens">Draft length of multi-token-prediction speculation (models with an MTP head); 0 disables it.</param>
/// <param name="SpeculationMaxBatch">Speculate only while at most this many sequences decode.</param>
/// <param name="MmprojPath">Default vision projector.</param>
/// <param name="ChatTemplatePath">Chat template file replacing the models' own.</param>
/// <param name="DefaultTemperature">Temperature for requests that give none.</param>
/// <param name="VramReserveBytes">Device memory kept free; null defers to the config file, then the built-in default.</param>
/// <param name="VramBudgetBytes">Total device memory cap; null defers to the config file.</param>
/// <param name="MaxLoadedModels">Resident model limit; null defers to the config file.</param>
/// <param name="Lazy">Load models on first request.</param>
/// <param name="IdleTimeoutSeconds">Idle unload timeout; null defers to the config file.</param>
/// <param name="StoreRoot">The model store that model names are looked up in.</param>
/// <param name="Pull">How models missing from the store are pulled.</param>
internal sealed record ServeSettings(
    IReadOnlyList<string> PositionalModels,
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
    long? KvCacheTokens,
    int NgramDraftTokens,
    int MtpDraftTokens,
    int SpeculationMaxBatch,
    string? MmprojPath,
    string? ChatTemplatePath,
    float DefaultTemperature,
    long? VramReserveBytes,
    long? VramBudgetBytes,
    int? MaxLoadedModels,
    bool Lazy,
    double? IdleTimeoutSeconds,
    string StoreRoot,
    PullSettings Pull);
