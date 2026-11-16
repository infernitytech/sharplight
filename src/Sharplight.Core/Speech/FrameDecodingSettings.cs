using System.Text.Json;
using Sharplight.Models;

namespace Sharplight.Speech;

/// <summary>
/// The configuration of a frame-aligned speech decoder, read from the decoder component's configuration: <c>decoding</c> ("ctc", "rnnt"
/// or "tdt"), <c>blank</c> (the blank token id), <c>rowFrames</c> (feature frames per encoder row), and for transducers <c>durations</c>
/// (TDT: the row advances its duration logits stand for), <c>maxSymbolsPerStep</c> and <c>layers</c> (prediction network layers).
/// </summary>
internal sealed record FrameDecodingSettings
{
    /// <summary>The decoding names this class reads.</summary>
    public static readonly string[] Decodings = ["ctc", "rnnt", "tdt"];

    /// <summary>"ctc", "rnnt" or "tdt".</summary>
    public required string Decoding { get; init; }

    /// <summary>The blank token id.</summary>
    public required int Blank { get; init; }

    /// <summary>Feature frames per encoder row.</summary>
    public required int RowFrames { get; init; }

    /// <summary>TDT: the row advance of each duration logit; empty otherwise.</summary>
    public int[] Durations { get; init; } = [];

    /// <summary>Transducers: the most tokens emitted at one row before the row advances.</summary>
    public int MaxSymbolsPerStep { get; init; } = 10;

    /// <summary>Transducers: the layers of the prediction network.</summary>
    public int Layers { get; init; } = 1;

    /// <summary>True for CTC.</summary>
    public bool IsCtc => Decoding == "ctc";

    /// <summary>True when <paramref name="decoder"/> configures a frame-aligned decoding.</summary>
    /// <param name="decoder">The decoder component.</param>
    public static bool IsPresent(ModelComponent decoder) => Decodings.Contains(decoder.ConfigValue("decoding", ""));

    /// <summary>Reads the settings of <paramref name="decoder"/>.</summary>
    /// <param name="decoder">The decoder component.</param>
    /// <exception cref="InvalidDataException">The configuration is incomplete.</exception>
    public static FrameDecodingSettings From(ModelComponent decoder)
    {
        string decoding = decoder.ConfigValue("decoding", "");
        if (!Decodings.Contains(decoding))
        {
            throw new InvalidDataException($"Unknown frame-aligned speech decoding '{decoding}'.");
        }

        int[] durations = decoder.Config.TryGetValue("durations", out var value) && value.ValueKind == JsonValueKind.Array
            ? [.. value.EnumerateArray().Select(e => e.GetInt32())]
            : [];
        return new FrameDecodingSettings
        {
            Decoding = decoding,
            Blank = decoder.ConfigValue("blank", -1) is int blank and >= 0 ? blank : throw new InvalidDataException("The speech decoder has no blank token."),
            RowFrames = Math.Max(decoder.ConfigValue("rowFrames", 1), 1),
            Durations = durations,
            MaxSymbolsPerStep = Math.Max(decoder.ConfigValue("maxSymbolsPerStep", 10), 1),
            Layers = Math.Max(decoder.ConfigValue("layers", 1), 1),
        };
    }
}
