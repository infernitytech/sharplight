using Sharplight.Models;

namespace Sharplight.ImageGeneration;

/// <summary>
/// The training noise schedule of a diffusion model and how its inference timesteps are spaced, read from the model's
/// <c>scheduler</c> component (<see cref="DiffusionConfigKeys"/>). Computes the cumulative alphas the samplers work with, in the
/// precision the reference implementation uses (f32 betas, cumulative product accumulated in f32).
/// </summary>
public sealed record NoiseSchedule
{
    /// <summary>Training timesteps.</summary>
    public int TrainSteps { get; init; } = 1000;

    /// <summary>First beta.</summary>
    public float BetaStart { get; init; } = 0.00085f;

    /// <summary>Last beta.</summary>
    public float BetaEnd { get; init; } = 0.012f;

    /// <summary><c>linear</c>, <c>scaled_linear</c> or <c>squaredcos_cap_v2</c>.</summary>
    public string BetaSchedule { get; init; } = "scaled_linear";

    /// <summary>
    /// What the denoiser predicts: <c>epsilon</c>, <c>v_prediction</c> or <c>sample</c> (variance-preserving schedules of the betas), or
    /// <c>flow</c> (rectified flow: the velocity ε − x₀ of x = (1 − σ) · x₀ + σ · ε, σ from 1 to 0; the betas are unused).
    /// </summary>
    public string PredictionType { get; init; } = "epsilon";

    /// <summary>
    /// Spacing of the inference timesteps (<c>leading</c>, <c>trailing</c> or <c>linspace</c>), or <see langword="null"/> for each
    /// sampler's own default (as the reference implementation does for configurations that do not name one).
    /// </summary>
    public string? TimestepSpacing { get; init; }

    /// <summary>Offset added to <c>leading</c> timesteps.</summary>
    public int StepsOffset { get; init; }

    /// <summary>
    /// Whether DDIM-type samplers end at alpha 1 instead of the first training alpha, or <see langword="null"/> for the sampler's
    /// default (as in diffusers: DDIM on, PNDM off).
    /// </summary>
    public bool? SetAlphaToOne { get; init; }

    /// <summary>Whether the betas are rescaled to zero terminal SNR.</summary>
    public bool ZeroTerminalSnr { get; init; }

    /// <summary>Whether DDIM clips its clean-sample estimate to [-1, 1], or <see langword="null"/> for the sampler's default (on).</summary>
    public bool? ClipSample { get; init; }

    /// <summary>The sampler the model was published with (<see cref="Samplers"/>).</summary>
    public string Sampler { get; init; } = Samplers.Euler;

    /// <summary>Flow schedules: the static shift s of the noise levels, σ' = s·σ / (1 + (s − 1)·σ) (1: unshifted).</summary>
    public float Shift { get; init; } = 1f;

    /// <summary>
    /// Flow schedules: whether the shift depends on the image size, σ' = e^μ / (e^μ + 1/σ − 1) with μ interpolated linearly from
    /// <see cref="BaseShift"/> at <see cref="BaseSequenceLength"/> image tokens to <see cref="MaxShift"/> at <see cref="MaxSequenceLength"/>.
    /// </summary>
    public bool DynamicShift { get; init; }

    /// <summary>Flow schedules: μ at <see cref="BaseSequenceLength"/> image tokens.</summary>
    public double BaseShift { get; init; } = 0.5;

    /// <summary>Flow schedules: μ at <see cref="MaxSequenceLength"/> image tokens.</summary>
    public double MaxShift { get; init; } = 1.15;

    /// <summary>Flow schedules: the image token count of <see cref="BaseShift"/>.</summary>
    public int BaseSequenceLength { get; init; } = 256;

    /// <summary>Flow schedules: the image token count of <see cref="MaxShift"/>.</summary>
    public int MaxSequenceLength { get; init; } = 4096;

    /// <summary>
    /// Flow schedules: the range the noise levels are spaced over before the shift: <c>unit</c> (σ from 1 to 1 / steps, as the FLUX,
    /// Chroma and Qwen-Image pipelines pass them; the default) or <c>trained</c> (from the scheduler's largest to its smallest training
    /// noise level, diffusers' <c>FlowMatchEulerDiscreteScheduler.set_timesteps</c> without explicit sigmas, as the SD3 pipelines use it).
    /// </summary>
    public string FlowSigmaRange { get; init; } = "unit";

    /// <summary>Whether this is a rectified-flow schedule (<see cref="PredictionType"/> <c>flow</c>).</summary>
    public bool IsFlow => PredictionType == "flow";

    /// <summary>Reads the schedule from a <c>scheduler</c> component's configuration; absent keys keep the defaults.</summary>
    /// <param name="component">The scheduler component.</param>
    public static NoiseSchedule FromComponent(ModelComponent component)
    {
        ArgumentNullException.ThrowIfNull(component);
        var defaults = new NoiseSchedule();
        return new NoiseSchedule
        {
            TrainSteps = component.ConfigValue(DiffusionConfigKeys.TrainSteps, defaults.TrainSteps),
            BetaStart = component.ConfigValue(DiffusionConfigKeys.BetaStart, defaults.BetaStart),
            BetaEnd = component.ConfigValue(DiffusionConfigKeys.BetaEnd, defaults.BetaEnd),
            BetaSchedule = component.ConfigValue(DiffusionConfigKeys.BetaSchedule, defaults.BetaSchedule),
            PredictionType = component.ConfigValue(DiffusionConfigKeys.PredictionType, defaults.PredictionType),
            TimestepSpacing = component.ConfigValue<string?>(DiffusionConfigKeys.TimestepSpacing, null),
            StepsOffset = component.ConfigValue(DiffusionConfigKeys.StepsOffset, defaults.StepsOffset),
            SetAlphaToOne = component.Config.ContainsKey(DiffusionConfigKeys.SetAlphaToOne) ? component.ConfigValue(DiffusionConfigKeys.SetAlphaToOne, false) : null,
            ZeroTerminalSnr = component.ConfigValue(DiffusionConfigKeys.ZeroTerminalSnr, defaults.ZeroTerminalSnr),
            Sampler = component.ConfigValue(DiffusionConfigKeys.Sampler, defaults.Sampler),
            ClipSample = component.Config.ContainsKey(DiffusionConfigKeys.ClipSample) ? component.ConfigValue(DiffusionConfigKeys.ClipSample, true) : null,
            Shift = component.ConfigValue(DiffusionConfigKeys.Shift, defaults.Shift),
            DynamicShift = component.ConfigValue(DiffusionConfigKeys.DynamicShift, defaults.DynamicShift),
            BaseShift = component.ConfigValue(DiffusionConfigKeys.BaseShift, defaults.BaseShift),
            MaxShift = component.ConfigValue(DiffusionConfigKeys.MaxShift, defaults.MaxShift),
            BaseSequenceLength = component.ConfigValue(DiffusionConfigKeys.BaseSequenceLength, defaults.BaseSequenceLength),
            MaxSequenceLength = component.ConfigValue(DiffusionConfigKeys.MaxSequenceLength, defaults.MaxSequenceLength),
            FlowSigmaRange = component.ConfigValue(DiffusionConfigKeys.FlowSigmaRange, defaults.FlowSigmaRange),
        };
    }

    /// <summary>The cumulative products of the alphas (1 - beta) over the training timesteps.</summary>
    /// <exception cref="NotSupportedException">The beta schedule is unknown.</exception>
    public float[] AlphasCumulative()
    {
        var betas = Betas();
        var cumulative = new float[TrainSteps];
        float product = 1f;
        for (int t = 0; t < TrainSteps; t++)
        {
            product *= 1f - betas[t];
            cumulative[t] = product;
        }

        if (ZeroTerminalSnr)
        {
            RescaleToZeroTerminalSnr(cumulative);
        }

        return cumulative;
    }

    /// <summary>The noise levels sqrt((1 - ᾱ) / ᾱ) of the training timesteps (the "sigmas" of the Karras formulation).</summary>
    public float[] Sigmas()
    {
        var cumulative = AlphasCumulative();
        var sigmas = new float[cumulative.Length];
        for (int t = 0; t < sigmas.Length; t++)
        {
            sigmas[t] = MathF.Sqrt((1f - cumulative[t]) / cumulative[t]);
        }

        return sigmas;
    }

    private float[] Betas()
    {
        var betas = new float[TrainSteps];
        switch (BetaSchedule)
        {
            case "linear":
                for (int t = 0; t < TrainSteps; t++)
                {
                    betas[t] = (float)Lerp(BetaStart, BetaEnd, t);
                }

                break;
            case "scaled_linear":
                for (int t = 0; t < TrainSteps; t++)
                {
                    float root = (float)Lerp(MathF.Sqrt(BetaStart), MathF.Sqrt(BetaEnd), t);
                    betas[t] = root * root;
                }

                break;
            case "squaredcos_cap_v2":
                for (int t = 0; t < TrainSteps; t++)
                {
                    double next = AlphaBar((t + 1.0) / TrainSteps);
                    double current = AlphaBar((double)t / TrainSteps);
                    betas[t] = (float)Math.Min(1 - next / current, 0.999);
                }

                break;
            default:
                throw new NotSupportedException($"Beta schedule '{BetaSchedule}' is not supported (linear, scaled_linear, squaredcos_cap_v2).");
        }

        return betas;
    }

    private double Lerp(double first, double last, int t) => TrainSteps == 1 ? first : first + (last - first) * t / (TrainSteps - 1);

    private static double AlphaBar(double t) => Math.Pow(Math.Cos((t + 0.008) / 1.008 * Math.PI / 2), 2);

    /// <summary>Rescales the schedule so the last timestep has zero signal (Lin et al. 2023), keeping the first one.</summary>
    private static void RescaleToZeroTerminalSnr(float[] cumulative)
    {
        var roots = cumulative.Select(a => Math.Sqrt(a)).ToArray();
        double first = roots[0];
        double last = roots[^1];
        for (int t = 0; t < roots.Length; t++)
        {
            double shifted = (roots[t] - last) * first / (first - last);
            cumulative[t] = (float)(shifted * shifted);
        }
    }
}
