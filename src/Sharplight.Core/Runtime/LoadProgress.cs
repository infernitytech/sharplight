namespace Sharplight.Runtime;

/// <summary>
/// One progress report of loading or compiling a model (see <see cref="CompileOptions.Progress"/>). Within a stage
/// <see cref="Completed"/> grows towards <see cref="Total"/>, which is known from the stage's first report; a graph split over
/// several devices adds each further segment's work to the stage's <see cref="Total"/> when that segment starts. Providers
/// may report stages concurrently (e.g. a GPU provider uploads weights while kernels compile), and reports may arrive on
/// worker threads.
/// </summary>
/// <param name="Stage">The stage reported.</param>
/// <param name="Completed">Units of the stage done: steps, kernels or operations (<see cref="LoadStage.Compiling"/>), or bytes (<see cref="LoadStage.Uploading"/>).</param>
/// <param name="Total">Units of the stage known so far; equals <see cref="Completed"/> when the stage is done.</param>
/// <param name="Detail">Short human-readable context (graph name, device), or <see langword="null"/>.</param>
public readonly record struct LoadProgress(LoadStage Stage, long Completed, long Total, string? Detail = null)
{
    /// <summary>
    /// For <see cref="LoadStage.Compiling"/>: how many of the <see cref="Total"/> kernels came from a cache (model artifacts or
    /// the per-user kernel cache) instead of being compiled; 0 for other stages and providers without kernel caches.
    /// </summary>
    public long Cached { get; init; }

    /// <summary>Completed share of the stage, 0 to 1 (1 for a stage without work).</summary>
    public double Fraction => Total > 0 ? Math.Clamp((double)Completed / Total, 0, 1) : 1;

    /// <summary>True when the stage's known work is done (<see cref="Completed"/> reached <see cref="Total"/>).</summary>
    public bool IsComplete => Completed >= Total;
}
