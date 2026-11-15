using System.Text.Json;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// The noise schedule and default sampler of a diffusion model, translated from a diffusers <c>scheduler_config.json</c> (or the
/// family defaults of a checkpoint without one) into the canonical <c>latent_diffusion.scheduler.*</c> keys, which the adapter puts
/// into the model's <c>scheduler</c> component.
/// </summary>
internal sealed record SchedulerSettings
{
    /// <summary>Training timesteps.</summary>
    public int TrainSteps { get; init; } = 1000;

    /// <summary>First beta.</summary>
    public float BetaStart { get; init; } = 0.00085f;

    /// <summary>Last beta.</summary>
    public float BetaEnd { get; init; } = 0.012f;

    /// <summary>Beta schedule: <c>linear</c>, <c>scaled_linear</c> or <c>squaredcos_cap_v2</c>.</summary>
    public string BetaSchedule { get; init; } = "scaled_linear";

    /// <summary>What the denoiser predicts: <c>epsilon</c>, <c>v_prediction</c> or <c>sample</c>.</summary>
    public string PredictionType { get; init; } = "epsilon";

    /// <summary>
    /// How inference timesteps are spaced (<c>leading</c>, <c>trailing</c> or <c>linspace</c>), or <see langword="null"/> when the
    /// configuration leaves it to each sampler's default.
    /// </summary>
    public string? TimestepSpacing { get; init; }

    /// <summary>Offset added to <c>leading</c> timesteps.</summary>
    public int StepsOffset { get; init; } = 1;

    /// <summary>Whether DDIM's final step goes to alpha 1 (instead of the first training alpha).</summary>
    public bool SetAlphaToOne { get; init; }

    /// <summary>Whether the betas are rescaled to zero terminal SNR.</summary>
    public bool ZeroTerminalSnr { get; init; }

    /// <summary>The sampler the model was published with (Sharplight sampler name, e.g. <c>pndm</c>, <c>euler</c>, <c>dpm++2m-karras</c>).</summary>
    public string Sampler { get; init; } = "pndm";

    /// <summary>The original configuration file's text, kept as an attachment, or <see langword="null"/>.</summary>
    public string? SourceJson { get; init; }

    /// <summary>Writes the keys under <c>{prefix}scheduler.</c>.</summary>
    /// <param name="metadata">The canonical metadata.</param>
    /// <param name="prefix">The architecture prefix (<c>latent_diffusion.</c>).</param>
    public void WriteTo(IDictionary<string, object> metadata, string prefix)
    {
        string p = prefix + "scheduler.";
        metadata[p + "train_steps"] = TrainSteps;
        metadata[p + "beta_start"] = BetaStart;
        metadata[p + "beta_end"] = BetaEnd;
        metadata[p + "beta_schedule"] = BetaSchedule;
        metadata[p + "prediction_type"] = PredictionType;
        if (TimestepSpacing is not null)
        {
            metadata[p + "timestep_spacing"] = TimestepSpacing;
        }

        metadata[p + "steps_offset"] = StepsOffset;
        metadata[p + "set_alpha_to_one"] = SetAlphaToOne;
        metadata[p + "zero_terminal_snr"] = ZeroTerminalSnr;
        metadata[p + "sampler"] = Sampler;
        if (SourceJson is not null)
        {
            metadata[p + "source_config"] = SourceJson;
        }
    }

    /// <summary>Reads a diffusers <c>scheduler_config.json</c>.</summary>
    /// <param name="json">The file's text.</param>
    public static SchedulerSettings FromDiffusers(string json)
    {
        using var document = JsonDocument.Parse(json);
        var config = document.RootElement;
        string type = config.String("_class_name", "PNDMScheduler");
        bool karras = config.Bool("use_karras_sigmas", false);
        string sampler = type switch
        {
            "DDIMScheduler" => "ddim",
            "EulerDiscreteScheduler" => "euler",
            "EulerAncestralDiscreteScheduler" => "euler-ancestral",
            "DPMSolverMultistepScheduler" => karras ? "dpm++2m-karras" : "dpm++2m",
            "LCMScheduler" => "lcm",
            _ => "pndm",
        };
        return new SchedulerSettings
        {
            TrainSteps = config.Int("num_train_timesteps", 1000),
            BetaStart = config.Float("beta_start", 0.00085f),
            BetaEnd = config.Float("beta_end", 0.012f),
            BetaSchedule = config.String("beta_schedule", "scaled_linear"),
            PredictionType = config.String("prediction_type", "epsilon"),
            TimestepSpacing = config.TryGetProperty("timestep_spacing", out var spacing) && spacing.ValueKind == JsonValueKind.String ? spacing.GetString() : null,
            StepsOffset = config.Int("steps_offset", 0),
            SetAlphaToOne = config.Bool("set_alpha_to_one", false),
            ZeroTerminalSnr = config.Bool("rescale_betas_zero_snr", false),
            Sampler = sampler,
            SourceJson = json,
        };
    }
}
