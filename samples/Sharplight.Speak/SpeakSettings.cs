namespace Sharplight.Speak;

/// <summary>The command line of the speak sample.</summary>
/// <param name="ModelPath">The speech-synthesis model: an <c>.slm</c> package or the language-model file.</param>
/// <param name="VocoderPath">The codec decoder file that goes with a language-model file, or <see langword="null"/> for a package.</param>
/// <param name="Text">The text to speak.</param>
/// <param name="OutputPath">The WAV file to write.</param>
/// <param name="Voice">The voice name, or <see langword="null"/> for the default.</param>
/// <param name="UseCpu">Run on the CPU instead of the GPU.</param>
/// <param name="DeviceIndex">The device within the provider.</param>
/// <param name="Seed">Seed of the token sampling, or <see langword="null"/>.</param>
/// <param name="Greedy">Pick the most likely token every time.</param>
/// <param name="Language">The language code of the text (Kokoro: <c>a</c> American, <c>b</c> British English, <c>phonemes</c> for phoneme input), or <see langword="null"/>.</param>
/// <param name="Speed">The speaking rate (1 is natural).</param>
/// <param name="Stream">Write the audio as it is decoded, one segment at a time, and report the time to the first segment.</param>
/// <param name="ReferencePath">A WAV recording of the voice to speak with (F5-TTS), or <see langword="null"/>.</param>
/// <param name="ReferenceText">The words of the reference recording, or <see langword="null"/>.</param>
internal sealed record SpeakSettings(string ModelPath, string? VocoderPath, string Text, string OutputPath, string? Voice, bool UseCpu, int DeviceIndex, int? Seed,
    bool Greedy, string? Language, float Speed, bool Stream, string? ReferencePath = null, string? ReferenceText = null)
{
    /// <summary>The usage text.</summary>
    public const string Usage = """
        usage: sharplight-speak --model <file> [--vocoder <file>] --text "<text>" [--out speech.wav] [--voice <name>] [--cpu] [--device <n>] [--seed <n>] [--greedy] [--language <code>] [--speed <x>] [--stream] [--reference <wav> --reference-text "<words>"]

          --model     speech-synthesis model: an .slm package, a checkpoint folder, or the language-model GGUF of a codec language model (OuteTTS)
          --vocoder   the codec decoder (WavTokenizer GGUF, SNAC or Vocos folder) for a model distributed without it
          --reference a recording of the voice to speak with (F5-TTS);  --reference-text  its words
          --text      the text to speak
          --out       output WAV file (default speech.wav)
          --voice     voice of the model (default: the model's own)
          --cpu       use the CPU provider instead of a GPU
          --device    device index within the provider (default 0)
          --seed      sampling seed;  --greedy  always take the most likely token
          --language  language of the text (Kokoro: a, b or phonemes);  --speed  speaking rate (default 1)
          --stream    decode and write segment by segment
        """;

    /// <summary>Parses the arguments; returns <see langword="null"/> and prints the usage when they are invalid.</summary>
    /// <param name="args">The arguments.</param>
    public static SpeakSettings? Parse(string[] args)
    {
        string? model = null;
        string? vocoder = null;
        string? text = null;
        string? voice = null;
        string output = "speech.wav";
        bool cpu = false;
        bool greedy = false;
        bool stream = false;
        string? language = null;
        float speed = 1f;
        int device = 0;
        int? seed = null;
        string? reference = null;
        string? referenceText = null;
        for (int i = 0; i < args.Length; i++)
        {
            string value() => i + 1 < args.Length ? args[++i] : throw new ArgumentException($"{args[i]} needs a value.");
            switch (args[i])
            {
                case "--model":
                    model = value();
                    break;
                case "--vocoder":
                    vocoder = value();
                    break;
                case "--text":
                    text = value();
                    break;
                case "--out":
                    output = value();
                    break;
                case "--voice":
                    voice = value();
                    break;
                case "--device":
                    device = int.Parse(value(), System.Globalization.CultureInfo.InvariantCulture);
                    break;
                case "--seed":
                    seed = int.Parse(value(), System.Globalization.CultureInfo.InvariantCulture);
                    break;
                case "--cpu":
                    cpu = true;
                    break;
                case "--greedy":
                    greedy = true;
                    break;
                case "--language":
                    language = value();
                    break;
                case "--speed":
                    speed = float.Parse(value(), System.Globalization.CultureInfo.InvariantCulture);
                    break;
                case "--stream":
                    stream = true;
                    break;
                case "--reference":
                    reference = value();
                    break;
                case "--reference-text":
                    referenceText = value();
                    break;
                default:
                    throw new ArgumentException($"Unknown argument {args[i]}.");
            }
        }

        if (model is null || text is null)
        {
            return null;
        }

        return new SpeakSettings(model, vocoder, text, output, voice, cpu, device, seed, greedy, language, speed, stream, reference, referenceText);
    }
}
