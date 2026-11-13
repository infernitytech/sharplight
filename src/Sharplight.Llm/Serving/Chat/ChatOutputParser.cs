using System.Text;

namespace Sharplight.Llm.Serving.Chat;

public enum PieceKind { Content, Reasoning, ToolCall }
public sealed record ParsedPiece(PieceKind Kind, string Text = "", ToolCall? Call = null);

/// <summary>
/// Incremental parser of model output: separates reasoning (<c>&lt;think&gt;…&lt;/think&gt;</c>) from content and extracts tool calls
/// in any of the given <see cref="IToolCallFormat"/>s. Text that may be the start of a marker is held back until it is clear.
/// Blocks that fail to parse as tool calls are returned as plain content.
/// </summary>
public sealed class ChatOutputParser
{
    private enum Mode { Content, Reasoning, Tool }

    private readonly IReadOnlyList<IToolCallFormat> _formats;
    private readonly HashSet<string> _toolNames;
    private readonly string _thinkOpen, _thinkClose;
    private readonly List<string> _contentMarkers;
    private readonly IToolCallFormat? _bare;
    private readonly StringBuilder _buf = new();
    private Mode _mode;
    private IToolCallFormat? _cur;
    private bool _curBare;
    private bool _atStart = true;        // no non-whitespace content emitted yet
    private bool _trimLeading;           // drop leading whitespace of the next emitted text

    public int ToolCallCount { get; private set; }

    /// <param name="formats">Tool-call formats to recognise; empty/null to disable tool parsing.</param>
    /// <param name="toolNames">Names of the offered tools; calls to other names are left as text.</param>
    /// <param name="startInReasoning">The prompt already opened a think block (the output starts with reasoning).</param>
    public ChatOutputParser(IReadOnlyList<IToolCallFormat>? formats, IEnumerable<string>? toolNames, bool startInReasoning = false,
        string thinkOpen = "<think>", string thinkClose = "</think>")
    {
        _formats = formats ?? [];
        _toolNames = [.. toolNames ?? []];
        _thinkOpen = thinkOpen; _thinkClose = thinkClose;
        _contentMarkers = [thinkOpen, .. _formats.Where(f => f.StartMarker is not null).Select(f => f.StartMarker!)];
        _bare = _formats.FirstOrDefault(f => f.AllowLeadingJson);
        if (startInReasoning) { _mode = Mode.Reasoning; _trimLeading = true; }
    }

    public List<ParsedPiece> Push(string text)
    {
        var o = new List<ParsedPiece>();
        _buf.Append(text);
        Drain(false, o);
        return o;
    }

    public List<ParsedPiece> Finish()
    {
        var o = new List<ParsedPiece>();
        Drain(true, o);
        return o;
    }

    private bool HasTools => _formats.Count > 0 && _toolNames.Count > 0;

    private void Drain(bool final, List<ParsedPiece> o)
    {
        while (true)
        {
            string s = _buf.ToString();
            switch (_mode)
            {
                case Mode.Content:
                {
                    if (HasTools && _bare is not null && _atStart)
                    {
                        var t = s.TrimStart();
                        if (t.Length == 0 && !final)
                            {
                                return;
                            }

                            if (t.StartsWith('{')) { _mode = Mode.Tool; _cur = _bare; _curBare = true; _buf.Clear().Append(t); continue; }
                    }
                    var markers = HasTools ? _contentMarkers : _contentMarkers.GetRange(0, 1);
                    var (idx, m) = MarkerScanner.FindFirst(s, markers);
                    if (idx >= 0)
                    {
                        bool isThink = m == 0;
                        EmitContent(isThink ? s[..idx] : s[..idx].TrimEnd(), o);
                        _buf.Clear().Append(s, idx + markers[m].Length, s.Length - idx - markers[m].Length);
                        if (isThink) { _mode = Mode.Reasoning; _trimLeading = true; }
                        else { _mode = Mode.Tool; _cur = _formats.First(f => f.StartMarker == markers[m]); _curBare = false; }
                        continue;
                    }
                    int hold = final ? 0 : MarkerScanner.PartialSuffix(s, markers);
                    // With tools, whitespace in front of a (possible) tool-call marker is held back so it can be trimmed.
                    if (HasTools && !final)
                        {
                            hold += s.AsSpan(0, s.Length - hold).Length - s.AsSpan(0, s.Length - hold).TrimEnd().Length;
                        }

                        EmitContent(s[..(s.Length - hold)], o);
                    _buf.Clear().Append(s, s.Length - hold, hold);
                    return;
                }
                case Mode.Reasoning:
                {
                    int idx = s.IndexOf(_thinkClose, StringComparison.Ordinal);
                    if (idx >= 0)
                    {
                        EmitReasoning(s[..idx].TrimEnd(), o);
                        _buf.Clear().Append(s, idx + _thinkClose.Length, s.Length - idx - _thinkClose.Length);
                        _mode = Mode.Content; _trimLeading = true;
                        continue;
                    }
                    int hold = final ? 0 : MarkerScanner.PartialSuffix(s, [_thinkClose]);
                    string emit = s[..(s.Length - hold)];
                    if (!final) { int ws = emit.Length - emit.TrimEnd().Length; hold += ws; emit = emit[..(emit.Length - ws)]; }
                    else
                        {
                            emit = emit.TrimEnd();
                        }

                        EmitReasoning(emit, o);
                    _buf.Clear().Append(s, s.Length - hold, hold);
                    return;
                }
                default:
                {
                    var end = _cur!.EndMarker;
                    if (end is not null)
                    {
                        int idx = s.IndexOf(end, StringComparison.Ordinal);
                        if (idx < 0) { if (final) { Finish(s, "", o); _buf.Clear(); } return; }
                        Finish(s[..idx], end, o);
                        _buf.Clear().Append(s, idx + end.Length, s.Length - idx - end.Length);
                        continue;
                    }
                    if (!final)
                        {
                            return;
                        }

                        Finish(s, "", o);
                    _buf.Clear();
                    return;
                }
            }
        }
    }

    /// <summary>Completes a tool block: parsed calls become pieces, anything else is returned as content.</summary>
    private void Finish(string block, string endMarker, List<ParsedPiece> o)
    {
        var fmt = _cur!;
        _mode = Mode.Content;
        if (fmt.TryParse(block, out var calls) && calls.All(c => _toolNames.Contains(c.Name)))
        {
            foreach (var c in calls)
            {
                ToolCallCount++;
                o.Add(new ParsedPiece(PieceKind.ToolCall, Call: new ToolCall(NewId(), c.Name, c.ArgumentsJson)));
            }
            _atStart = false;
            _trimLeading = true;
        }
        else
        {
            _atStart = false;
            string raw = (_curBare ? "" : fmt.StartMarker) + block + endMarker;
            if (raw.Length > 0)
            {
                o.Add(new ParsedPiece(PieceKind.Content, raw));
            }
        }
        _curBare = false;
    }

    private string NewId() => "call_" + Guid.NewGuid().ToString("N")[..24];

    private void EmitContent(string text, List<ParsedPiece> o)
    {
        if (_trimLeading) { text = text.TrimStart(); if (text.Length > 0)
            {
                _trimLeading = false;
            }
        }
        if (text.Length == 0)
        {
            return;
        }

        if (text.AsSpan().Trim().Length > 0)
        {
            _atStart = false;
        }

        o.Add(new ParsedPiece(PieceKind.Content, text));
    }

    private void EmitReasoning(string text, List<ParsedPiece> o)
    {
        if (_trimLeading) { text = text.TrimStart(); if (text.Length > 0)
            {
                _trimLeading = false;
            }
        }
        if (text.Length > 0)
        {
            o.Add(new ParsedPiece(PieceKind.Reasoning, text));
        }
    }
}
