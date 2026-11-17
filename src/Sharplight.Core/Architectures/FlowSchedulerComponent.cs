using System.Text;
using Sharplight.ImageGeneration;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// The scheduler component of rectified-flow image generators (FLUX, SD3, Chroma, Qwen-Image): the flow-matching keys under
/// <c>{prefix}scheduler.*</c> (<c>train_steps</c>, <c>shift</c>, <c>dynamic_shift</c>, <c>base_shift</c>, <c>max_shift</c>,
/// <c>base_sequence_length</c>, <c>max_sequence_length</c>, <c>sampler</c>, <c>timestep_spacing</c> (<c>scheduler</c>: the scheduler's own levels, see <see cref="FlowMatchEulerSampler"/>), <c>source_config</c>) as a description component, and the
/// original scheduler configuration as an attachment.
/// </summary>
internal static class FlowSchedulerComponent
{
    /// <summary>The attachment name of the original scheduler configuration.</summary>
    public const string Attachment = "scheduler/source-config.json";

    /// <summary>Adds the original scheduler configuration, if the source has one, to <paramref name="attachments"/>.</summary>
    /// <param name="src">The canonical source.</param>
    /// <param name="prefix">The architecture prefix (<c>flux.</c>).</param>
    /// <param name="attachments">The definition's attachments.</param>
    /// <returns>Whether an attachment was added.</returns>
    public static bool AddAttachment(ModelSource src, string prefix, Dictionary<string, IWeightSource> attachments)
    {
        if (src.Get<string?>(prefix + "scheduler.source_config", null) is not { } schedulerJson)
        {
            return false;
        }

        byte[] bytes = Encoding.UTF8.GetBytes(schedulerJson);
        attachments[Attachment] = WeightInfo.Derived(Attachment, DataType.U8, new Shape(bytes.Length), () => bytes).Source;
        return true;
    }

    /// <summary>The scheduler component.</summary>
    /// <param name="src">The canonical source.</param>
    /// <param name="prefix">The architecture prefix.</param>
    /// <param name="hasAttachment">Whether <see cref="AddAttachment"/> added the original configuration.</param>
    /// <param name="defaultShift">The static shift when the source has none.</param>
    public static ModelComponent Create(ModelSource src, string prefix, bool hasAttachment, float defaultShift = 1f)
    {
        string s = prefix + "scheduler.";
        var config = new ComponentConfigBuilder()
            .Add(DiffusionConfigKeys.TrainSteps, src.Get(s + "train_steps", 1000))
            .Add(DiffusionConfigKeys.PredictionType, "flow")
            .Add(DiffusionConfigKeys.Shift, src.Get(s + "shift", defaultShift))
            .Add(DiffusionConfigKeys.DynamicShift, src.Get(s + "dynamic_shift", false))
            .Add(DiffusionConfigKeys.BaseShift, src.Get(s + "base_shift", 0.5))
            .Add(DiffusionConfigKeys.MaxShift, src.Get(s + "max_shift", 1.15))
            .Add(DiffusionConfigKeys.BaseSequenceLength, src.Get(s + "base_sequence_length", 256))
            .Add(DiffusionConfigKeys.MaxSequenceLength, src.Get(s + "max_sequence_length", 4096))
            .Add(DiffusionConfigKeys.Sampler, src.Get(s + "sampler", Samplers.Euler));
        if (src.Get(s + "shift_terminal", 0.0) is > 0 and var terminal)
        {
            config.Add(DiffusionConfigKeys.ShiftTerminal, terminal);
        }

        if (src.Get(s + "empirical_shift", false))
        {
            config.Add(DiffusionConfigKeys.EmpiricalShift, true);
        }

        if (src.Has(s + "sample_sigmas"))
        {
            config.Add(DiffusionConfigKeys.SampleSigmas, src.Get<float[]>(s + "sample_sigmas"));
        }

        if (src.Get<string?>(s + "timestep_spacing", null) is { } spacing)
        {
            config.Add(DiffusionConfigKeys.TimestepSpacing, spacing);
        }

        return new ModelComponent
        {
            Role = ModelComponentRoles.Scheduler,
            Config = config.Build(),
            Attachments = hasAttachment
                ? new Dictionary<string, string> { ["source-config"] = Attachment }
                : new Dictionary<string, string>(),
        };
    }
}
