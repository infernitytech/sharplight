using System.CommandLine;
using Sharplight.Samples;
using Sharplight.Speech;

namespace Sharplight.Transcribe;

/// <summary>The command line of <c>sharplight-transcribe</c>.</summary>
public sealed class TranscribeOptionSet
{
    /// <summary>The model argument.</summary>
    public Argument<string> Model { get; } = new("model") { Description = "Speech-recognition model: .slm package, Hugging Face directory or whisper.cpp .bin file." };

    /// <summary>The audio argument.</summary>
    public Argument<string> Audio { get; } = new("audio") { Description = "WAV file (any sample rate and channel count; resampled to the model's rate)." };

    /// <summary><c>--cpu</c>.</summary>
    public Option<bool> Cpu { get; } = new("--cpu") { Description = "Run on the CPU provider instead of a GPU." };

    /// <summary><c>--device</c>.</summary>
    public Option<int> Device { get; } = OptionFactory.Integer("--device", "Index of the device to use (see sharplight --devices).", 0, minimum: 0, helpName: "index");

    /// <summary><c>--language</c>.</summary>
    public Option<string?> Language { get; } = new("--language", "-l") { Description = "Spoken language (ISO 639-1, e.g. en); detected when omitted.", HelpName = "code" };

    /// <summary><c>--translate</c>.</summary>
    public Option<bool> Translate { get; } = new("--translate") { Description = "Translate the speech into English (multilingual models)." };

    /// <summary><c>--no-timestamps</c>.</summary>
    public Option<bool> NoTimestamps { get; } = new("--no-timestamps") { Description = "Decode without timestamp tokens (one segment per 30 s window)." };

    /// <summary><c>--prompt</c>.</summary>
    public Option<string?> Prompt { get; } = new("--prompt") { Description = "Text that conditions the decoder (names, spelling, style).", HelpName = "text" };

    /// <summary><c>--condition-on-previous-text</c>.</summary>
    public Option<bool> ConditionOnPreviousText { get; } = new("--condition-on-previous-text")
    {
        Description = "Prompt each 30 s window with the text recognized so far (long recordings).",
    };

    /// <summary><c>--format</c>.</summary>
    public Option<TranscriptFormat> Format { get; } = new("--format", "-f")
    {
        Description = "Output format: text, segments, srt or json.",
        DefaultValueFactory = _ => TranscriptFormat.Text,
    };

    /// <summary><c>--verbose</c>.</summary>
    public Option<bool> Verbose { get; } = new("--verbose", "-v") { Description = "Print load and compile messages to standard error." };

    /// <summary>Builds the root command running <paramref name="run"/>.</summary>
    /// <param name="run">Runs the command with the parsed settings.</param>
    public RootCommand BuildCommand(Func<TranscribeSettings, CancellationToken, Task<int>> run)
    {
        var command = new RootCommand("Transcribes a WAV file with a speech-recognition model (Whisper and compatible).");
        command.Arguments.Add(Model);
        command.Arguments.Add(Audio);
        foreach (Option option in new Option[] { Cpu, Device, Language, Translate, NoTimestamps, Prompt, ConditionOnPreviousText, Format, Verbose })
        {
            command.Options.Add(option);
        }

        command.SetAction(CommandRunner.Guard((parseResult, cancellationToken) => run(Read(parseResult), cancellationToken)));
        return command;
    }

    private TranscribeSettings Read(ParseResult parseResult)
    {
        var recognition = new SpeechRecognitionOptions
        {
            Language = parseResult.GetValue(Language),
            Translate = parseResult.GetValue(Translate),
            Timestamps = parseResult.GetValue(NoTimestamps) ? TimestampGranularity.None : TimestampGranularity.Segment,
            Prompt = parseResult.GetValue(Prompt),
            ConditionOnPreviousText = parseResult.GetValue(ConditionOnPreviousText),
        };
        return new TranscribeSettings(parseResult.GetValue(Model)!, parseResult.GetValue(Audio)!, parseResult.GetValue(Cpu), parseResult.GetValue(Device),
            recognition, parseResult.GetValue(Format), parseResult.GetValue(Verbose));
    }
}
