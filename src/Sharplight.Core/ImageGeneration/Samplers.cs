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

    /// <summary>The samplers of rectified-flow models (<see cref="NoiseSchedule.IsFlow"/>): Euler on the flow ODE.</summary>
    public static IReadOnlyList<string> Flow { get; } = [Euler];

    /// <summary>Creates a sampler for one run.</summary>
    /// <param name="name">A name of <see cref="All"/> (case-insensitive; <c>_</c> and <c>-</c> are interchangeable); for flow schedules one of <see cref="Flow"/>.</param>
    /// <param name="schedule">The model's noise schedule.</param>
    /// <param name="steps">Denoising steps.</param>
    /// <param name="imageTokens">The denoiser's image tokens per image (flow schedules with a dynamic shift).</param>
    /// <exception cref="NotSupportedException">The name is unknown or does not apply to the schedule.</exception>
    internal static DiffusionSampler Create(string name, NoiseSchedule schedule, int steps, int imageTokens = 0)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(steps, 1);
        if (schedule.IsFlow)
        {
            return Normalize(name) switch
            {
                Euler or "flow-euler" => new FlowMatchEulerSampler(schedule, steps, imageTokens),
                _ => throw new NotSupportedException($"Sampler '{name}' does not apply to rectified-flow models (supported: {string.Join(", ", Flow)})."),
            };
        }

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
