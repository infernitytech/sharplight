using Sharplight.Audio;
using Sharplight.Llm;

namespace Sharplight.Speech;

/// <summary>Per-request settings of <see cref="ISpeechSynthesizer"/>; implementations ignore what their model cannot do.</summary>
public sealed record SpeechSynthesisOptions
{
    /// <summary>The voice (<see cref="ISpeechSynthesizer.Voices"/>), or <see langword="null"/> for the model's default voice.</summary>
    public string? Voice { get; init; }

    /// <summary>Language of the text (ISO 639-1, e.g. <c>en</c>), or <see langword="null"/> to let the model decide.</summary>
    public string? Language { get; init; }

    /// <summary>Speaking rate relative to the model's natural rate (1 = unchanged).</summary>
    public float Speed { get; init; } = 1f;

    /// <summary>
    /// Token selection of models that generate codec or acoustic tokens autoregressively. Left at its defaults (only <see cref="SamplingOptions.Seed"/>
    /// set, or nothing), the model's own sampling settings apply with that seed.
    /// </summary>
    public SamplingOptions Sampling { get; init; } = new();

    /// <summary>
    /// A recording of the voice to speak with (voice cloning), for models that take one (F5-TTS needs it); <see langword="null"/> for the
    /// <see cref="Voice"/>. Any rate and channel count.
    /// </summary>
    public AudioBuffer? ReferenceAudio { get; init; }

    /// <summary>
    /// A saved voice (<see cref="IVoiceCloner.CreateVoiceAsync"/>, <see cref="VoiceLibrary"/>) to speak with instead of <see cref="Voice"/> and
    /// <see cref="ReferenceAudio"/>; its model family must be the model's. Its words serve as <see cref="ReferenceText"/> when that is not set.
    /// </summary>
    public VoiceProfile? VoiceProfile { get; init; }

    /// <summary>The words spoken in <see cref="ReferenceAudio"/>, for models that align the reference with its text (F5-TTS).</summary>
    public string? ReferenceText { get; init; }

    /// <summary>
    /// A natural-language instruction on how to speak (dialect, emotion, speed), for models that take one (CosyVoice 3); <see langword="null"/> for none.
    /// </summary>
    public string? Instructions { get; init; }

    /// <summary>
    /// Spells numbers, dates, times, currency amounts, percentages and abbreviations as words before the text reaches the model (<see cref="Normalization.TextNormalizers"/>
    /// for EN, DE, FR, ES, IT, PT, ZH and JA), for the models that benefit from it; <see langword="false"/> passes the text as written.
    /// </summary>
    public bool NormalizeText { get; init; } = true;

    /// <summary>
    /// The normalizer to use instead of the built-in one of the request's <see cref="Language"/> (or of the text's script when none is set), for every model
    /// that normalizes; <see langword="null"/> for the default.
    /// </summary>
    public Normalization.ITextNormalizer? TextNormalizer { get; init; }

    /// <summary>
    /// The voices of the speakers of a multi-speaker script (speaker label as written in the script, for example <c>1</c> for <c>Speaker 1:</c>, to the saved voice), for models that
    /// speak several voices in one request (VibeVoice); a speaker without an entry gets the model's own voice. <see cref="ReferenceAudio"/> and <see cref="VoiceProfile"/> serve the first speaker.
    /// </summary>
    public IReadOnlyDictionary<string, VoiceProfile>? SpeakerVoices { get; init; }

    /// <summary>Denoising steps per frame of models whose acoustic features come from a diffusion head (VibeVoice), or <see langword="null"/> for the model's default; more steps cost more time.</summary>
    public int? DiffusionSteps { get; init; }

    /// <summary>Classifier-free guidance scale of models with a diffusion head (VibeVoice), or <see langword="null"/> for the model's default; 1 turns guidance off.</summary>
    public float? GuidanceScale { get; init; }

    /// <summary>The defaults.</summary>
    public static SpeechSynthesisOptions Default { get; } = new();
}
