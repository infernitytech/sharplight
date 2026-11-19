using Sharplight.Models;

namespace Sharplight.Audio;

/// <summary>
/// The time schedule of the rectified-flow audio models (<see cref="TimedAudioDiffusionGraphs.ScheduleKey"/>): <c>steps + 1</c> times from 1 (noise) to 0 (data), a
/// linear ramp warped by the model's distribution shift for the effective latent length, computed in f32 as the stable-audio-3 package does
/// (<c>torch.linspace</c>'s ramp, the first time forced back to 1).
/// </summary>
internal sealed class FlowTimeSchedule
{
    private readonly string _type;
    private readonly ModelComponent _denoiser;

    /// <summary>Reads the schedule of a denoiser component.</summary>
    /// <param name="denoiser">The denoiser component.</param>
    public FlowTimeSchedule(ModelComponent denoiser)
    {
        _denoiser = denoiser;
        _type = denoiser.ConfigValue(TimedAudioDiffusionGraphs.ScheduleKey, "none");
        if (_type is not ("none" or "logsnr" or "flux" or "full"))
        {
            throw new NotSupportedException($"The time schedule '{_type}' is not supported.");
        }
    }

    /// <summary>The times of a run.</summary>
    /// <param name="steps">The number of steps.</param>
    /// <param name="length">The effective latent length (the frames of the requested duration).</param>
    /// <returns><c>steps + 1</c> times from 1 to 0.</returns>
    public float[] Times(int steps, int length)
    {
        var t = Linspace(1f, 0f, steps + 1);
        float seq = Math.Max(length, 1);
        for (int i = 0; i < t.Length; i++)
        {
            t[i] = Shift(t[i], seq);
        }

        t[0] = 1f;
        return t;
    }

    // torch.linspace in f32: the first half counts up from the start, the second half down from the end.
    private static float[] Linspace(float start, float end, int count)
    {
        var values = new float[count];
        if (count == 1)
        {
            values[0] = start;
            return values;
        }

        float step = (end - start) / (count - 1);
        int halfway = count / 2;
        for (int i = 0; i < count; i++)
        {
            values[i] = i < halfway ? start + (step * i) : end - (step * (count - i - 1));
        }

        return values;
    }

    private float Shift(float t, float seq)
    {
        switch (_type)
        {
            case "logsnr":
            {
                float anchor = Value(TimedAudioDiffusionGraphs.AnchorLogSnrKey, -6.2f);
                float end = Value(TimedAudioDiffusionGraphs.LogSnrEndKey, 2f);
                float rate = Value(TimedAudioDiffusionGraphs.ShiftRateKey, 1f);
                float anchorLength = Value(TimedAudioDiffusionGraphs.AnchorLengthKey, 2000f);
                float start = anchor - (rate * MathF.Log2(seq / anchorLength));
                if (t <= 0f)
                {
                    return 0f;
                }

                if (t >= 1f)
                {
                    return 1f;
                }

                float logsnr = end - (t * (end - start));
                return 1f / (1f + MathF.Exp(logsnr));
            }

            case "flux":
            {
                float minLength = Value(TimedAudioDiffusionGraphs.MinLengthKey, 256f);
                float maxLength = Value(TimedAudioDiffusionGraphs.MaxLengthKey, 4096f);
                float logMin = MathF.Log(Math.Max(Value(TimedAudioDiffusionGraphs.AlphaMinKey, 1f), 1e-8f));
                float logMax = MathF.Log(Math.Max(Value(TimedAudioDiffusionGraphs.AlphaMaxKey, 1f), 1e-8f));
                float logMinSeq = MathF.Log(minLength);
                float logMaxSeq = MathF.Log(maxLength);
                if (logMaxSeq == logMinSeq)
                {
                    logMaxSeq += 1e-8f;
                }

                float frac = (MathF.Log(Math.Clamp(seq, minLength, maxLength)) - logMinSeq) / (logMaxSeq - logMinSeq);
                float alpha = MathF.Exp(logMin + (frac * (logMax - logMin)));
                return alpha * t / (1f + ((alpha - 1f) * t));
            }

            case "full":
            {
                float minLength = Value(TimedAudioDiffusionGraphs.MinLengthKey, 256f);
                float maxLength = Value(TimedAudioDiffusionGraphs.MaxLengthKey, 4096f);
                float baseShift = Value(TimedAudioDiffusionGraphs.BaseShiftKey, 0.5f);
                float maxShift = Value(TimedAudioDiffusionGraphs.MaxShiftKey, 1.15f);
                float mu = -(baseShift + ((maxShift - baseShift) * (Math.Clamp(seq, minLength, maxLength) - minLength) / (maxLength - minLength)));
                return 1f - (MathF.Exp(mu) / (MathF.Exp(mu) + ((1f / (1f - t)) - 1f)));
            }

            default:
                return t;
        }
    }

    private float Value(string key, float fallback) => (float)_denoiser.ConfigValue(key, (double)fallback);
}
