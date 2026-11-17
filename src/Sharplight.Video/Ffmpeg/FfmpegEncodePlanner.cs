using System.Globalization;

namespace Sharplight.Video.Ffmpeg;

/// <summary>Validates the public <see cref="VideoEncodeOptions"/> against a format and turns them into an <see cref="FfmpegEncodePlan"/>.</summary>
internal static class FfmpegEncodePlanner
{
    /// <summary>Builds the plan.</summary>
    /// <param name="container">The output format.</param>
    /// <param name="codecs">The codecs available for it on this machine, the default first.</param>
    /// <param name="clip">The clip.</param>
    /// <param name="options">The options.</param>
    /// <param name="seekableOutput">Whether the destination can seek (MP4 / MOV are fragmented otherwise).</param>
    /// <exception cref="ArgumentException">An option names something the format does not offer.</exception>
    public static FfmpegEncodePlan Create(FfmpegContainer container, IReadOnlyList<FfmpegCodec> codecs, VideoClip clip, VideoEncodeOptions options, bool seekableOutput)
    {
        FfmpegCodec codec = PickCodec(container, codecs, options.Codec);
        string pixelFormat = Pick(options.PixelFormat, codec.PixelFormats, codec.PixelFormats[0], $"pixel format of {codec.Id}");
        var encoderOptions = new Dictionary<string, string>(codec.FixedOptions, StringComparer.Ordinal);
        if (codec.Presets.Count > 0)
        {
            string preset = Pick(options.Preset, [.. codec.Presets.Select(p => p.Id)], codec.DefaultPreset ?? codec.Presets[0].Id, $"preset of {codec.Id}");
            foreach ((string key, string value) in codec.Presets.First(p => p.Id == preset).Options)
            {
                encoderOptions[key] = value;
            }
        }

        double rate = options.FramesPerSecond ?? clip.FramesPerSecond;
        if (!double.IsFinite(rate) || rate <= 0)
        {
            throw new ArgumentException("The frame rate must be positive and finite.", nameof(options));
        }

        (int numerator, int denominator) = FrameRate.ToFraction(rate);
        int width = codec.NeedsEvenSize ? (clip.Width + 1) & ~1 : clip.Width;
        int height = codec.NeedsEvenSize ? (clip.Height + 1) & ~1 : clip.Height;
        int quality = Math.Clamp(options.Quality, 0, 100);
        long bitRate = options.BitrateKilobitsPerSecond is int kilobits && kilobits > 0 ? kilobits * 1000L : 0;
        int? rateFactor = RateFactor(codec, options, quality);
        var plan = new FfmpegEncodePlan
        {
            Container = container,
            Codec = codec,
            PixelFormat = pixelFormat,
            Width = width,
            Height = height,
            RateNumerator = numerator,
            RateDenominator = denominator,
            EncoderOptions = encoderOptions,
            MuxerOptions = MuxerOptions(container, options, seekableOutput),
            KeyframeInterval = options.KeyframeInterval is int interval && interval > 0 ? interval : null,
            Threads = options.MaxThreads > 0 ? options.MaxThreads : Math.Clamp(Environment.ProcessorCount, 1, 16),
            BitRate = bitRate,
            Audio = AudioPlan(container, clip, options),
        };

        if (bitRate > 0)
        {
            // Bitrate mode: the constant-quality switches of the fixed options (b=0) do not apply.
            encoderOptions.Remove("b");
            return plan;
        }

        string factor = rateFactor?.ToString(CultureInfo.InvariantCulture) ?? string.Empty;
        long pixelsPerSecond = (long)(width * (long)height * rate);
        switch (codec.Quality)
        {
            case QualityControl.Crf:
                encoderOptions["crf"] = factor;
                return codec.NeedsBitrateWithRateFactor ? plan with { BitRate = Math.Max(2_000_000, pixelsPerSecond) } : plan;
            case QualityControl.NvencConstantQuality:
                encoderOptions["cq"] = factor;
                return plan;
            case QualityControl.QuantizerWindow:
                // OpenH264 aims at a bitrate even in quality mode: a generous one leaves the QP window in control.
                return plan with
                {
                    QuantizerMinimum = Math.Max(1, rateFactor!.Value - 3),
                    QuantizerMaximum = Math.Min(51, rateFactor.Value + 3),
                    BitRate = Math.Max(4_000_000, pixelsPerSecond),
                };
            case QualityControl.Percent when codec.EncoderName.Contains("videotoolbox", StringComparison.Ordinal):
                return plan with { GlobalQualityPercent = Math.Max(1, quality) };
            case QualityControl.Percent:
                encoderOptions["quality"] = quality.ToString(CultureInfo.InvariantCulture);
                return plan;
            case QualityControl.CompressionLevel:
                return plan with { CompressionLevel = 1 + quality * 8 / 100 };
            case QualityControl.Dither:
                return plan with { Dither = quality >= 100 ? "none" : "sierra2_4a" };
            default:
                return plan;
        }
    }

    /// <summary>The FFmpeg encoder of an audio codec id.</summary>
    public static string AudioEncoderName(string codec) => codec switch
    {
        VideoCodecs.Aac => "aac",
        VideoCodecs.Opus => "libopus",
        VideoCodecs.Flac => "flac",
        VideoCodecs.Pcm => "pcm_s16le",
        _ => throw new ArgumentException($"Unknown audio codec '{codec}'.", nameof(codec)),
    };

    private static FfmpegCodec PickCodec(FfmpegContainer container, IReadOnlyList<FfmpegCodec> codecs, string? requested)
    {
        if (requested is null)
        {
            return codecs.Count > 0 ? codecs[0] : throw new PlatformNotSupportedException($"No {container.Format} encoder is available.");
        }

        FfmpegCodec? codec = codecs.FirstOrDefault(c => string.Equals(c.Id, requested, StringComparison.OrdinalIgnoreCase));
        if (codec is not null)
        {
            return codec;
        }

        bool known = container.Codecs.Concat(container.Hardware).Any(c => string.Equals(c.Id, requested, StringComparison.OrdinalIgnoreCase));
        string reason = known ? "is not available on this machine" : $"is not a codec of {container.Format}";
        throw new ArgumentException($"The video codec '{requested}' {reason} (available: {string.Join(", ", codecs.Select(c => c.Id))}).", nameof(requested));
    }

    private static string Pick(string? requested, IReadOnlyList<string> allowed, string fallback, string what)
    {
        if (requested is null)
        {
            return fallback;
        }

        return allowed.FirstOrDefault(a => string.Equals(a, requested, StringComparison.OrdinalIgnoreCase))
            ?? throw new ArgumentException($"'{requested}' is not a {what} (choose one of {string.Join(", ", allowed)}).", nameof(requested));
    }

    private static int? RateFactor(FfmpegCodec codec, VideoEncodeOptions options, int quality)
    {
        if (codec.RateFactorRange is not (int minimum, int maximum))
        {
            return null;
        }

        if (options.ConstantRateFactor is int explicitFactor)
        {
            if (explicitFactor < minimum || explicitFactor > maximum)
            {
                throw new ArgumentException($"The rate factor of {codec.Id} must be in [{minimum}, {maximum}], got {explicitFactor}.", nameof(options));
            }

            return explicitFactor;
        }

        double factor = codec.WorstRateFactor + (codec.BestRateFactor - codec.WorstRateFactor) * quality / 100.0;
        return Math.Clamp((int)Math.Round(factor), minimum, maximum);
    }

    private static Dictionary<string, string> MuxerOptions(FfmpegContainer container, VideoEncodeOptions options, bool seekableOutput)
    {
        var muxer = new Dictionary<string, string>(StringComparer.Ordinal);
        if (container.Muxer is "mp4" or "mov" && !seekableOutput)
        {
            // The movie header first: a non-seekable destination cannot be patched at the end.
            muxer["movflags"] = "frag_keyframe+empty_moov+default_base_moof";
        }
        else if (container.Muxer == "gif")
        {
            muxer["loop"] = options.Loop ? "0" : "-1";
        }
        else if (container.Muxer == "apng")
        {
            muxer["plays"] = options.Loop ? "0" : "1";
        }

        return muxer;
    }

    private static FfmpegAudioPlan? AudioPlan(FfmpegContainer container, VideoClip clip, VideoEncodeOptions options)
    {
        if (!options.IncludeAudio || clip.Audio is null || container.AudioCodecs.Count == 0)
        {
            if (options.AudioCodec is not null && container.AudioCodecs.Count == 0)
            {
                throw new ArgumentException($"{container.Format} has no audio track.", nameof(options));
            }

            return null;
        }

        string codec = Pick(options.AudioCodec, container.AudioCodecs, container.AudioCodecs[0], $"audio codec of {container.Format}");
        long bitRate = options.AudioBitrateKilobitsPerSecond is int kilobits && kilobits > 0 ? kilobits * 1000L : 0;
        return new FfmpegAudioPlan(AudioEncoderName(codec), bitRate);
    }
}
