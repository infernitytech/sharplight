using System.Text;

namespace Sharplight.Llm.Serving.Chat;

/// <summary>
/// Incremental parser of chat model output: cuts the output at stop strings, separates reasoning (<c>&lt;think&gt;…&lt;/think&gt;</c>,
/// or harmony analysis messages) from content and extracts tool calls in any of the configured <see cref="IToolCallFormat"/>s.
/// Text that may be the start of a marker or stop string is held back until it is clear. Blocks that fail to parse as tool calls
/// are returned as plain content. Leading whitespace of each reasoning and content block is dropped, and so is trailing whitespace
/// in front of a marker and at the end of reasoning; the answer keeps trailing whitespace at the end of the output.
/// </summary>
/// <remarks>
/// This is the one output parser of every chat path (the OpenAI / Anthropic serving endpoints, Studio and the
/// Microsoft.Extensions.AI clients), so they all split the same text the same way. An instance parses one output; it is not
/// thread-safe.
/// </remarks>
public sealed class ChatOutputParser
{
    private enum Mode { Content, Reasoning, Tool }

    private readonly IReadOnlyList<IToolCallFormat> _formats;
    private readonly HashSet<string> _toolNames;
    private readonly string _reasoningEnd;
    private readonly string[] _stops;
    private readonly List<string> _contentMarkers;
    private readonly IToolCallFormat? _bare;
    private readonly StringBuilder _stopBuffer = new();
    private readonly StringBuilder _buffer = new();
    private readonly HarmonyOutputTranslator? _harmony;
    private Mode _mode;
    private IToolCallFormat? _current;
    private bool _currentBare;
    private bool _atStart = true;        // no non-whitespace content emitted yet
    private bool _trimLeading = true;    // drop leading whitespace of the next emitted text of the current block
    private string _pendingWhitespace = "";

    /// <summary>Creates a parser for one output.</summary>
    /// <param name="options">Markers, tool-call formats and stop strings; defaults (no tools, <c>&lt;think&gt;</c> markers) when null.</param>
    public ChatOutputParser(ChatOutputParserOptions? options = null)
    {
        options ??= new ChatOutputParserOptions();
        _harmony = options.Harmony ? new HarmonyOutputTranslator() : null;
        _formats = options.ToolCallFormats;
        _toolNames = [.. options.ToolNames];
        _reasoningEnd = options.ReasoningEnd;
        _stops = [.. options.StopStrings.Where(stop => stop.Length > 0)];
        _contentMarkers = [options.ReasoningStart, .. _formats.Where(f => f.StartMarker is not null).Select(f => f.StartMarker!)];
        _bare = _formats.FirstOrDefault(f => f.AllowLeadingJson);
        _mode = options.StartsInReasoning ? Mode.Reasoning : Mode.Content;
    }

    /// <summary>Number of tool calls returned so far.</summary>
    public int ToolCallCount { get; private set; }

    /// <summary>True once a stop string was seen: the output ended there and further input is ignored.</summary>
    public bool Stopped { get; private set; }

    /// <summary>
    /// True when a rendered chat prompt ends inside an open reasoning block (the template's generation prompt opened it, as Qwen3,
    /// DeepSeek-R1 and similar templates do with thinking on), so the output starts with reasoning. A prompt whose last reasoning
    /// block is closed (thinking off: <c>&lt;think&gt;\n\n&lt;/think&gt;\n\n</c>) does not.
    /// </summary>
    /// <param name="prompt">The rendered prompt text.</param>
    /// <param name="reasoningStart">Marker that opens a reasoning block.</param>
    /// <returns>Whether the output starts in reasoning.</returns>
    public static bool PromptEndsInReasoning(string prompt, string reasoningStart = ChatOutputParserOptions.DefaultReasoningStart)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        return prompt.TrimEnd().EndsWith(reasoningStart, StringComparison.Ordinal);
    }

    /// <summary>True when a chat template writes OpenAI harmony messages (gpt-oss): its output needs <see cref="ChatOutputParserOptions.Harmony"/>.</summary>
    /// <param name="template">The chat template source, or null.</param>
    /// <returns>Whether the template is a harmony template.</returns>
    public static bool IsHarmonyTemplate(string? template) => template?.Contains("<|channel|>", StringComparison.Ordinal) == true;

    /// <summary>Parses the next piece of output.</summary>
    /// <param name="text">The text generated since the previous call.</param>
    /// <returns>The pieces that are final so far (possibly none).</returns>
    public IReadOnlyList<ChatOutputPiece> Push(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        var output = new List<ChatOutputPiece>();
        if (Stopped || text.Length == 0)
        {
            return output;
        }

        Accept(CutAtStop(text), output, final: false);
        return output;
    }

    /// <summary>Ends the output: flushes held-back text and completes an unterminated tool call.</summary>
    /// <returns>The remaining pieces.</returns>
    public IReadOnlyList<ChatOutputPiece> Complete()
    {
        var output = new List<ChatOutputPiece>();
        string rest = _stopBuffer.ToString();
        _stopBuffer.Clear();
        Accept(rest, output, final: true);
        return output;
    }

    /// <summary>Applies the stop strings: returns the text that is certainly before any stop string.</summary>
    private string CutAtStop(string text)
    {
        if (_stops.Length == 0)
        {
            return text;
        }

        _stopBuffer.Append(text);
        string s = _stopBuffer.ToString();
        var (cut, _) = MarkerScanner.FindFirst(s, _stops);
        if (cut >= 0)
        {
            Stopped = true;
            _stopBuffer.Clear();
            return s[..cut];
        }

        int hold = MarkerScanner.PartialSuffix(s, _stops);
        _stopBuffer.Remove(0, s.Length - hold);
        return s[..(s.Length - hold)];
    }

    private void Accept(string text, List<ChatOutputPiece> output, bool final)
    {
        if (_harmony is null)
        {
            _buffer.Append(text);
        }
        else
        {
            _buffer.Append(_harmony.Push(text));
            if (final)
            {
                _buffer.Append(_harmony.Finish());
            }
        }

        while (Drain(final, output))
        {
            // Each pass consumes a marker; loop until the buffer holds no complete marker.
        }
    }

    private bool HasTools => _formats.Count > 0 && _toolNames.Count > 0;

    /// <summary>Consumes the buffer in the current mode. Returns true when a marker was consumed and the buffer needs another pass.</summary>
    private bool Drain(bool final, List<ChatOutputPiece> output) => _mode switch
    {
        Mode.Content => DrainContent(final, output),
        Mode.Reasoning => DrainReasoning(final, output),
        _ => DrainTool(final, output),
    };

    /// <summary>Content mode: emits text up to the next reasoning or tool-call marker.</summary>
    private bool DrainContent(bool final, List<ChatOutputPiece> output)
    {
        string s = _buffer.ToString();
        if (HasTools && _bare is not null && _atStart)
        {
            var trimmed = s.TrimStart();
            if (trimmed.Length == 0 && !final)
            {
                return false;
            }

            if (trimmed.StartsWith('{'))
            {
                Enter(Mode.Tool);
                _current = _bare;
                _currentBare = true;
                _buffer.Clear().Append(trimmed);
                return true;
            }
        }

        var markers = HasTools ? _contentMarkers : _contentMarkers.GetRange(0, 1);
        var (index, marker) = MarkerScanner.FindFirst(s, markers);
        if (index >= 0)
        {
            Emit(ChatOutputPieceKind.Content, s[..index], blockEnds: true, output);
            _buffer.Clear().Append(s, index + markers[marker].Length, s.Length - index - markers[marker].Length);
            if (marker == 0)
            {
                Enter(Mode.Reasoning);
            }
            else
            {
                Enter(Mode.Tool);
                _current = _formats.First(f => f.StartMarker == markers[marker]);
                _currentBare = false;
            }

            return true;
        }

        int hold = final ? 0 : MarkerScanner.PartialSuffix(s, markers);
        Emit(ChatOutputPieceKind.Content, s[..(s.Length - hold)], blockEnds: false, output);
        _buffer.Clear().Append(s, s.Length - hold, hold);
        if (final && !_trimLeading && _pendingWhitespace.Length > 0)
        {
            // The answer's own trailing whitespace (for example before a stop string) is part of it.
            output.Add(new ChatOutputPiece(ChatOutputPieceKind.Content, _pendingWhitespace));
            _pendingWhitespace = "";
        }

        return false;
    }

    /// <summary>Reasoning mode: emits text up to the reasoning end marker.</summary>
    private bool DrainReasoning(bool final, List<ChatOutputPiece> output)
    {
        string s = _buffer.ToString();
        int index = s.IndexOf(_reasoningEnd, StringComparison.Ordinal);
        if (index >= 0)
        {
            Emit(ChatOutputPieceKind.Reasoning, s[..index], blockEnds: true, output);
            _buffer.Clear().Append(s, index + _reasoningEnd.Length, s.Length - index - _reasoningEnd.Length);
            Enter(Mode.Content);
            return true;
        }

        int hold = final ? 0 : MarkerScanner.PartialSuffix(s, [_reasoningEnd]);
        Emit(ChatOutputPieceKind.Reasoning, s[..(s.Length - hold)], blockEnds: final, output);
        _buffer.Clear().Append(s, s.Length - hold, hold);
        return false;
    }

    /// <summary>Tool mode: collects the block up to the format's end marker (or the end of the output when it has none).</summary>
    private bool DrainTool(bool final, List<ChatOutputPiece> output)
    {
        string s = _buffer.ToString();
        var end = _current!.EndMarker;
        int index = end is null ? -1 : s.IndexOf(end, StringComparison.Ordinal);
        if (index >= 0)
        {
            CompleteToolBlock(s[..index], end!, output);
            _buffer.Clear().Append(s, index + end!.Length, s.Length - index - end.Length);
            return true;
        }

        if (final)
        {
            CompleteToolBlock(s, "", output);
            _buffer.Clear();
        }

        return false;
    }

    private void Enter(Mode mode)
    {
        _mode = mode;
        _trimLeading = true;
        _pendingWhitespace = "";
    }

    /// <summary>Completes a tool block: parsed calls become pieces, anything else is returned as content.</summary>
    private void CompleteToolBlock(string block, string endMarker, List<ChatOutputPiece> output)
    {
        var format = _current!;
        bool bare = _currentBare;
        Enter(Mode.Content);
        _currentBare = false;
        _atStart = false;
        if (format.TryParse(block, out var calls) && calls.All(c => _toolNames.Contains(c.Name)))
        {
            foreach (var call in calls)
            {
                ToolCallCount++;
                output.Add(new ChatOutputPiece(ChatOutputPieceKind.ToolCall, Call: new ToolCall(NewId(), call.Name, call.ArgumentsJson)));
            }

            return;
        }

        string raw = (bare ? "" : format.StartMarker) + block + endMarker;
        if (raw.Length > 0)
        {
            output.Add(new ChatOutputPiece(ChatOutputPieceKind.Content, raw));
        }
    }

    private static string NewId() => "call_" + Guid.NewGuid().ToString("N")[..24];

    /// <summary>
    /// Emits text of the current block. Leading whitespace of a block is dropped; trailing whitespace is held until more text of the
    /// same block follows, so it is dropped when the block ends (<paramref name="blockEnds"/>).
    /// </summary>
    private void Emit(ChatOutputPieceKind kind, string text, bool blockEnds, List<ChatOutputPiece> output)
    {
        if (_trimLeading)
        {
            text = text.TrimStart();
            if (text.Length == 0)
            {
                return;
            }

            _trimLeading = false;
        }

        string body = text.TrimEnd();
        if (body.Length > 0)
        {
            output.Add(new ChatOutputPiece(kind, _pendingWhitespace + body));
            _pendingWhitespace = "";
            if (kind == ChatOutputPieceKind.Content)
            {
                _atStart = false;
            }
        }

        _pendingWhitespace = blockEnds ? "" : _pendingWhitespace + text[body.Length..];
    }
}
