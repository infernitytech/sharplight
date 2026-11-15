namespace Sharplight.Speech;

/// <summary>The progress of one recognition request across the windows of a recording.</summary>
/// <param name="options">The request settings.</param>
/// <param name="contentFrames">Feature frames that hold the recording (without the front end's padding).</param>
/// <param name="history">Tokens that prompt the next window (the request prompt, then previous text when conditioning on it).</param>
internal sealed class RecognitionState(SpeechRecognitionOptions options, int contentFrames, List<int> history)
{
    /// <summary>The request settings.</summary>
    public SpeechRecognitionOptions Options { get; } = options;

    /// <summary>Feature frames that hold the recording.</summary>
    public int ContentFrames { get; } = Math.Max(contentFrames, 1);

    /// <summary>Tokens that prompt the next window.</summary>
    public List<int> History { get; } = history;

    /// <summary>The first feature frame of the next window.</summary>
    public int Seek { get; set; }

    /// <summary>The language token of the request, once given or detected; null for single-language models.</summary>
    public int? Language { get; set; }
}
