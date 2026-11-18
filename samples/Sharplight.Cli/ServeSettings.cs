using Sharplight.Samples.Hub;
using Sharplight.Samples;

namespace Sharplight.Cli;

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
/// <param name="MinStepTokens">Smallest token budget of the load-adaptive step; null: the engine's default, 0: always the full budget.</param>
/// <param name="KvCacheTokens">Paged KV cache size in tokens shared by the slots; null: the default (every slot's full context, paged
/// with prefix caching where the device supports it); 0: the contiguous per-slot cache.</param>
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
/// <param name="OnDemand">Also serve every model of the store by name, loading it on its first request.</param>
/// <param name="StoreRoot">The model store that model names are looked up in.</param>
/// <param name="Pull">How models missing from the store are pulled.</param>
/// <param name="AllowClientCacheSalt">Accept the <c>cache_salt</c> request field from clients.</param>
/// <param name="NoPrefixCache">Turn prefix caching off (every prompt is computed in full).</param>
/// <param name="KvStateSnapshots">Recurrent-state snapshots for prefix caching of models with recurrent state; null: the default.</param>
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
    int? MinStepTokens,
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
    bool OnDemand,
    string StoreRoot,
    PullSettings Pull,
    bool AllowClientCacheSalt = false,
    bool NoPrefixCache = false,
    int? KvStateSnapshots = null);
