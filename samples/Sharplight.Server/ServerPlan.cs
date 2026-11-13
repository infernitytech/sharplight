namespace Sharplight.Server;

/// <summary>The command line merged with the configuration file: the models to serve and the host-wide limits.</summary>
/// <param name="Models">All models, in registration order; names are unique.</param>
/// <param name="ReserveBytes">Device memory kept free for other applications.</param>
/// <param name="BudgetBytes">Total device memory cap, or null for what the driver reports as available.</param>
/// <param name="MaxLoadedModels">Resident model limit; zero means unlimited.</param>
/// <param name="LazyByDefault">Models without their own <c>lazy</c> setting load on first request.</param>
/// <param name="IdleTimeoutSeconds">Idle unload timeout; null or zero disables it.</param>
internal sealed record ServerPlan(
    IReadOnlyList<ModelSpec> Models,
    long ReserveBytes,
    long? BudgetBytes,
    int MaxLoadedModels,
    bool LazyByDefault,
    double? IdleTimeoutSeconds);
