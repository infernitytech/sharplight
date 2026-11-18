namespace Sharplight.Studio.Services.Smoke;

/// <summary>The result of a smoke run: what <c>--smoke</c> writes as JSON for the CI workflow (<c>.github/workflows/studio-published-smoke.yml</c>).</summary>
internal sealed record SmokeReport
{
    /// <summary>The version of the running build (the assembly's informational version without build metadata).</summary>
    public required string Version { get; init; }

    /// <summary>The operating system's description.</summary>
    public required string Os { get; init; }

    /// <summary>The process architecture.</summary>
    public required string Architecture { get; init; }

    /// <summary>The storage mode in effect (<c>Portable</c>, <c>System</c> or <c>Custom</c>) and where.</summary>
    public required string Storage { get; init; }

    /// <summary>The devices models can be loaded on.</summary>
    public IReadOnlyList<string> Devices { get; init; } = [];

    /// <summary>The package that was installed and run, or null when the run stopped before.</summary>
    public string? Package { get; init; }

    /// <summary>The steps in order; a failed step is the last one.</summary>
    public IReadOnlyList<SmokeStep> Steps { get; init; } = [];

    /// <summary>Every step passed.</summary>
    public bool Passed => Steps.Count > 0 && Steps.All(step => step.Passed);

    /// <summary>The detail of the failed step, or null.</summary>
    public string? Error => Steps.FirstOrDefault(step => !step.Passed)?.Detail;

    /// <summary>Total time in seconds.</summary>
    public double Seconds { get; init; }
}
