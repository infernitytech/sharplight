using System.Text.RegularExpressions;

namespace Sharplight.Studio.ViewModels.Speech;

/// <summary>A multi-speaker script: lines <c>Speaker N: text</c> (a line without a label continues the turn before).</summary>
internal static partial class SpeechScript
{
    /// <summary>One turn of a script.</summary>
    /// <param name="Speaker">The speaker label as written (<c>1</c> for <c>Speaker 1:</c>).</param>
    /// <param name="Text">The words.</param>
    internal sealed record Turn(string Speaker, string Text);

    [GeneratedRegex(@"^\s*Speaker\s+([^\s:]+)\s*:\s*(.*)$", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant)]
    private static partial Regex TurnLine();

    /// <summary>The turns of a script; a script without labels is one turn of speaker 0.</summary>
    /// <param name="text">The script.</param>
    public static IReadOnlyList<Turn> Parse(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        var turns = new List<Turn>();
        foreach (string raw in text.Split('\n'))
        {
            string line = raw.TrimEnd('\r');
            if (TurnLine().Match(line) is { Success: true } match)
            {
                turns.Add(new Turn(match.Groups[1].Value, match.Groups[2].Value.Trim()));
            }
            else if (line.Trim().Length > 0)
            {
                if (turns.Count == 0)
                {
                    turns.Add(new Turn("0", line.Trim()));
                }
                else
                {
                    var last = turns[^1];
                    turns[^1] = last with { Text = (last.Text + " " + line.Trim()).Trim() };
                }
            }
        }

        return turns;
    }

    /// <summary>The distinct speaker labels of a script in order of first appearance; empty when no line carries a label.</summary>
    /// <param name="text">The script.</param>
    public static IReadOnlyList<string> Speakers(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        var labels = new List<string>();
        foreach (string line in text.Split('\n'))
        {
            if (TurnLine().Match(line.TrimEnd('\r')) is { Success: true } match && !labels.Contains(match.Groups[1].Value, StringComparer.Ordinal))
            {
                labels.Add(match.Groups[1].Value);
            }
        }

        return labels;
    }
}
