using System.Text;
using Microsoft.Extensions.AI;

namespace Sharplight.Extensions.AI;

/// <summary>
/// Incremental post-processing of generated text: trims stop sequences, splits <c>&lt;think&gt;</c> reasoning from the answer
/// and extracts tool calls. Feed text pieces with <see cref="Push"/>, then call <see cref="Complete"/>.
/// </summary>
public sealed class OutputParser
{
    private enum Mode { Text, Reasoning, ToolCall }

    private const string ThinkStart = "<think>", ThinkEnd = "</think>";

    private readonly string[] _stops;
    private readonly string? _toolStart, _toolEnd;
    private readonly IList<IToolCallParser> _parsers;
    private readonly ToolSchemas _tools;
    private readonly StringBuilder _stopBuf = new();
    private readonly StringBuilder _buf = new();
    private readonly StringBuilder _toolBody = new();
    private readonly StringBuilder _pendingWs = new();
    private Mode _mode;
    private bool _leading = true;

    /// <summary>True once a stop sequence was seen; further input is ignored.</summary>
    public bool Stopped { get; private set; }
    public int ToolCallCount { get; private set; }

    /// <param name="startInReasoning">The prompt already opened a <c>&lt;think&gt;</c> block (the output begins with reasoning).</param>
    /// <param name="stopSequences">Output is cut before the first occurrence of any of these.</param>
    /// <param name="toolStart">Tool-call start marker, or null to treat tool-call text as plain text.</param>
    public OutputParser(bool startInReasoning, IEnumerable<string>? stopSequences, string? toolStart, string? toolEnd,
        IList<IToolCallParser>? parsers, ToolSchemas? tools)
    {
        _mode = startInReasoning ? Mode.Reasoning : Mode.Text;
        _stops = stopSequences?.Where(s => s.Length > 0).ToArray() ?? [];
        _toolStart = toolStart;
        _toolEnd = toolEnd;
        _parsers = parsers ?? [];
        _tools = tools ?? ToolSchemas.Empty;
    }

    public IReadOnlyList<AIContent> Push(string text)
    {
        var output = new List<AIContent>();
        if (Stopped || text.Length == 0)
        {
            return output;
        }

        if (_stops.Length == 0) { Process(text, output, flush: false); return output; }
        _stopBuf.Append(text);
        var s = _stopBuf.ToString();
        int cut = -1;
        foreach (var stop in _stops)
        {
            int i = s.IndexOf(stop, StringComparison.Ordinal);
            if (i >= 0 && (cut < 0 || i < cut))
            {
                cut = i;
            }
        }
        if (cut >= 0)
        {
            Stopped = true;
            _stopBuf.Clear();
            Process(s[..cut], output, flush: false);
            return output;
        }
        int hold = HoldBack(s, _stops);
        _stopBuf.Remove(0, s.Length - hold);
        Process(s[..(s.Length - hold)], output, flush: false);
        return output;
    }

    /// <summary>Flushes held-back text at the end of the stream.</summary>
    public IReadOnlyList<AIContent> Complete()
    {
        var output = new List<AIContent>();
        if (!Stopped && _stopBuf.Length > 0)
        {
            var rest = _stopBuf.ToString();
            _stopBuf.Clear();
            Process(rest, output, flush: false);
        }
        Process("", output, flush: true);
        return output;
    }

    private void Process(string text, List<AIContent> output, bool flush)
    {
        _buf.Append(text);
        while (true)
        {
            var s = _buf.ToString();
            var markers = _mode switch
            {
                Mode.Text => _toolStart is null ? (ThinkStart, Marker.Think, null, Marker.None) : (ThinkStart, Marker.Think, _toolStart, Marker.Tool),
                Mode.Reasoning => (ThinkEnd, Marker.ThinkEnd, null, Marker.None),
                _ => (_toolEnd!, Marker.ToolEnd, null, Marker.None),
            };
            int i1 = markers.Item1.Length > 0 ? s.IndexOf(markers.Item1, StringComparison.Ordinal) : -1;
            int i2 = markers.Item3 is { } m2 ? s.IndexOf(m2, StringComparison.Ordinal) : -1;
            int idx; string marker; Marker kind;
            if (i1 >= 0 && (i2 < 0 || i1 <= i2)) { idx = i1; marker = markers.Item1; kind = markers.Item2; }
            else if (i2 >= 0) { idx = i2; marker = markers.Item3!; kind = markers.Item4; }
            else
            {
                // No complete marker: emit everything except a possible marker prefix at the end.
                int hold = flush ? 0 : HoldBack(s, markers.Item3 is null ? [markers.Item1] : [markers.Item1, markers.Item3]);
                Emit(s[..(s.Length - hold)], output);
                _buf.Remove(0, s.Length - hold);
                if (flush)
                {
                    FlushMode(output);
                }

                return;
            }
            Emit(s[..idx], output);
            _buf.Remove(0, idx + marker.Length);
            switch (kind)
            {
                case Marker.Think: SwitchTo(Mode.Reasoning); break;
                case Marker.ThinkEnd: SwitchTo(Mode.Text); break;
                case Marker.Tool: SwitchTo(Mode.ToolCall); break;
                case Marker.ToolEnd:
                    FinishToolCall(_toolBody.ToString(), output);
                    _toolBody.Clear();
                    SwitchTo(Mode.Text);
                    break;
            }
        }
    }

    private enum Marker { None, Think, ThinkEnd, Tool, ToolEnd }

    private void SwitchTo(Mode m)
    {
        _pendingWs.Clear(); // whitespace between blocks is dropped
        _leading = true;
        _mode = m;
    }

    private void Emit(string text, List<AIContent> output)
    {
        if (text.Length == 0)
        {
            return;
        }

        if (_mode == Mode.ToolCall) { _toolBody.Append(text); return; }
        // Leading whitespace of a block is dropped; trailing whitespace is held until more text follows.
        if (_leading)
        {
            text = text.TrimStart();
            if (text.Length == 0)
            {
                return;
            }

            _leading = false;
        }
        string trimmed = text.TrimEnd();
        string ws = text[trimmed.Length..];
        if (trimmed.Length > 0)
        {
            string body = _pendingWs.ToString() + trimmed;
            _pendingWs.Clear();
            output.Add(_mode == Mode.Reasoning ? new TextReasoningContent(body) : new TextContent(body));
        }
        _pendingWs.Append(ws);
    }

    private void FlushMode(List<AIContent> output)
    {
        if (_mode == Mode.ToolCall)
        {
            // Unterminated tool call (token limit): use it if it parses, otherwise surface it as text.
            var body = _toolBody.ToString();
            _toolBody.Clear();
            FinishToolCall(body, output, markUnparsedAsText: true);
        }
        _pendingWs.Clear();
    }

    private void FinishToolCall(string body, List<AIContent> output, bool markUnparsedAsText = false)
    {
        foreach (var p in _parsers)
        {
            if (p.TryParse(body, _tools) is { } call)
            {
                ToolCallCount++;
                output.Add(call);
                return;
            }
        }
        // Unknown format: keep the raw text so nothing is silently lost.
        output.Add(new TextContent(_toolStart + body + (markUnparsedAsText ? "" : _toolEnd)));
    }

    /// <summary>Length of the longest proper prefix of any candidate that is a suffix of <paramref name="s"/>.</summary>
    private static int HoldBack(string s, IEnumerable<string> candidates)
    {
        int best = 0;
        foreach (var c in candidates)
        {
            int max = Math.Min(c.Length - 1, s.Length);
            for (int len = max; len > best; len--)
            {
                if (s.AsSpan(s.Length - len).SequenceEqual(c.AsSpan(0, len))) { best = len; break; }
            }
        }
        return best;
    }
}
