namespace Sharplight.ImageGeneration;

/// <summary>
/// Names of the samplers (schedulers) the built-in latent-diffusion pipeline implements, for
/// <see cref="ImageGenerationOptions.Scheduler"/> and the model's configured default. Each follows the corresponding diffusers
/// scheduler step by step, so a run with the same initial noise reproduces it.
/// </summary>
public static class Samplers
{
    /// <summary>Euler (deterministic).</summary>
    public const string Euler = "euler";

    /// <summary>Euler with the Karras noise levels.</summary>
    public const string EulerKarras = "euler-karras";

    /// <summary>Ancestral Euler (stochastic: fresh noise every step).</summary>
    public const string EulerAncestral = "euler-ancestral";

    /// <summary>DDIM with η = 0 (deterministic).</summary>
    public const string Ddim = "ddim";

    /// <summary>PLMS / PNDM without the Runge-Kutta warm-up (the SD 1.x default).</summary>
    public const string Pndm = "pndm";

    /// <summary>DPM-Solver++(2M).</summary>
    public const string DpmPlusPlus2M = "dpm++2m";

    /// <summary>DPM-Solver++(2M) with the Karras noise levels.</summary>
    public const string DpmPlusPlus2MKarras = "dpm++2m-karras";

    /// <summary>Every supported name.</summary>
    public static IReadOnlyList<string> All { get; } = [Euler, EulerKarras, EulerAncestral, Ddim, Pndm, DpmPlusPlus2M, DpmPlusPlus2MKarras];

    /// <summary>Creates a sampler for one run.</summary>
    /// <param name="name">A name of <see cref="All"/> (case-insensitive; <c>_</c> and <c>-</c> are interchangeable).</param>
    /// <param name="schedule">The model's noise schedule.</param>
    /// <param name="steps">Denoising steps.</param>
    /// <exception cref="NotSupportedException">The name is unknown.</exception>
    internal static DiffusionSampler Create(string name, NoiseSchedule schedule, int steps)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(steps, 1);
        return Normalize(name) switch
        {
            Euler => new EulerSampler(schedule, steps, karras: false),
            EulerKarras => new EulerSampler(schedule, steps, karras: true),
            EulerAncestral or "euler-a" => new EulerAncestralSampler(schedule, steps),
            Ddim => new DdimSampler(schedule, steps),
            Pndm or "plms" => new PndmSampler(schedule, steps),
            DpmPlusPlus2M => new DpmSolverSampler(schedule, steps, karras: false),
            DpmPlusPlus2MKarras => new DpmSolverSampler(schedule, steps, karras: true),
            _ => throw new NotSupportedException($"Sampler '{name}' is not supported (supported: {string.Join(", ", All)})."),
        };
    }

    private static string Normalize(string name) => name.Trim().ToLowerInvariant().Replace('_', '-').Replace(' ', '-');
}
