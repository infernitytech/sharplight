using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace Sharplight.Llm.Serving.Chat;

/// <summary>
/// Streams OpenAI harmony output (gpt-oss) into the markup <see cref="ChatOutputParser"/> understands: an <c>analysis</c> message
/// becomes a <c>&lt;think&gt;…&lt;/think&gt;</c> block, a message addressed to <c>functions.NAME</c> (commentary channel) becomes a
/// Hermes <c>&lt;tool_call&gt;{"name": NAME, "arguments": …}&lt;/tool_call&gt;</c> block, and <c>final</c> (or any other) messages
/// become content. The message framing (<c>&lt;|start|&gt;role</c>, <c>&lt;|channel|&gt;…&lt;|message|&gt;</c>,
/// <c>&lt;|end|&gt;</c>, <c>&lt;|return|&gt;</c>, <c>&lt;|call|&gt;</c>) is removed. Text that may be the start of a marker is held
/// back until it is clear.
/// </summary>
internal sealed partial class HarmonyOutputTranslator
{
    private const string Message = "<|message|>";
    private const string FunctionsPrefix = "functions.";
    private static readonly string[] s_terminators = ["<|end|>", "<|return|>", "<|call|>", "<|start|>"];
    private static readonly string[] s_headerStarts = ["<|start|>", "<|channel|>"];

    private enum State { Header, Content, Reasoning, Tool }

    private readonly StringBuilder _in = new();
    private readonly StringBuilder _toolArguments = new();
    private State _state = State.Header;
    private string _toolName = "";

    [GeneratedRegex(@"to=(\S+?)(?=\s|<\||$)")]
    private static partial Regex Recipient();

    [GeneratedRegex(@"<\|channel\|>\s*(\w+)")]
    private static partial Regex Channel();

    /// <summary>Translates the next piece of output; returns the text that is final so far.</summary>
    public string Push(string text)
    {
        _in.Append(text);
        var output = new StringBuilder();
        Drain(final: false, output);
        return output.ToString();
    }

    /// <summary>Translates what is left at the end of the output, closing an open reasoning or tool block.</summary>
    public string Finish()
    {
        var output = new StringBuilder();
        Drain(final: true, output);
        switch (_state)
        {
            case State.Reasoning:
                output.Append("</think>");
                break;
            case State.Tool:
                AppendToolCall(output);
                break;
        }

        _state = State.Header;
        return output.ToString();
    }

    private void Drain(bool final, StringBuilder output)
    {
        while (_in.Length > 0)
        {
            string s = _in.ToString();
            if (_state == State.Header)
            {
                if (!TryReadHeader(s, final, output))
                {
                    return;
                }

                continue;
            }

            var (index, terminator) = FirstTerminator(s);
            if (index < 0)
            {
                int keep = final ? 0 : PartialMarkerLength(s);
                Emit(s[..^keep], output);
                _in.Remove(0, s.Length - keep);
                return;
            }

            Emit(s[..index], output);
            _in.Remove(0, index + (terminator == "<|start|>" ? 0 : terminator.Length));
            EndMessage(output);
        }
    }

    // Reads a message header up to <|message|>; output that does not start with a header is content. False = wait for more text.
    private bool TryReadHeader(string s, bool final, StringBuilder output)
    {
        string trimmed = s.TrimStart();
        if (trimmed.Length == 0)
        {
            if (final)
            {
                _in.Clear();
            }

            return false;
        }

        bool header = s_headerStarts.Any(h => trimmed.StartsWith(h, StringComparison.Ordinal));
        if (!header)
        {
            if (!final && s_headerStarts.Any(h => h.StartsWith(trimmed, StringComparison.Ordinal)))
            {
                return false;   // could still become a header marker
            }

            _state = State.Content;
            return true;
        }

        int message = s.IndexOf(Message, StringComparison.Ordinal);
        if (message < 0)
        {
            if (final)
            {
                _in.Clear();
            }

            return false;
        }

        string text = s[..message];
        _in.Remove(0, message + Message.Length);
        string channel = Channel().Match(text) is { Success: true } c ? c.Groups[1].Value : "final";
        string? recipient = Recipient().Match(text) is { Success: true } r ? r.Groups[1].Value : null;
        if (recipient is not null && recipient.StartsWith(FunctionsPrefix, StringComparison.Ordinal))
        {
            _state = State.Tool;
            _toolName = recipient[FunctionsPrefix.Length..];
            _toolArguments.Clear();
        }
        else if (channel == "analysis")
        {
            _state = State.Reasoning;
            output.Append("<think>");
        }
        else
        {
            _state = State.Content;
        }

        return true;
    }

    private void Emit(string text, StringBuilder output)
    {
        if (_state == State.Tool)
        {
            _toolArguments.Append(text);
        }
        else
        {
            output.Append(text);
        }
    }

    private void EndMessage(StringBuilder output)
    {
        switch (_state)
        {
            case State.Reasoning:
                output.Append("</think>");
                break;
            case State.Tool:
                AppendToolCall(output);
                break;
        }

        _state = State.Header;
    }

    private void AppendToolCall(StringBuilder output)
    {
        string arguments = _toolArguments.ToString().Trim();
        output.Append("<tool_call>{\"name\": ").Append(JsonSerializer.Serialize(_toolName)).Append(", \"arguments\": ")
            .Append(arguments.Length == 0 ? "{}" : arguments).Append("}</tool_call>");
        _toolArguments.Clear();
    }

    private static (int Index, string Terminator) FirstTerminator(string s)
    {
        int best = -1;
        string found = "";
        foreach (string terminator in s_terminators)
        {
            int i = s.IndexOf(terminator, StringComparison.Ordinal);
            if (i >= 0 && (best < 0 || i < best))
            {
                best = i;
                found = terminator;
            }
        }

        return (best, found);
    }

    // Length of the longest suffix of s that is a proper prefix of a terminator.
    private static int PartialMarkerLength(string s)
    {
        int keep = 0;
        foreach (string terminator in s_terminators)
        {
            for (int n = Math.Min(terminator.Length - 1, s.Length); n > keep; n--)
            {
                if (s.EndsWith(terminator[..n], StringComparison.Ordinal))
                {
                    keep = n;
                    break;
                }
            }
        }

        return keep;
    }
}
