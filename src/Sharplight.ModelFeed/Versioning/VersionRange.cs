using System.Diagnostics.CodeAnalysis;

namespace Sharplight.ModelFeed.Versioning;

/// <summary>
/// A version range in one of two notations: NuGet intervals (<c>[1.0.0,2.0.0)</c>, <c>(,2.0.0]</c>, <c>[1.2.0]</c> exactly, and a bare
/// <c>1.0.0</c> meaning "1.0.0 or higher") or comparator sets (<c>&gt;=1.0.0</c>, <c>&gt;=1.0.0 &lt;2.0.0</c>, <c>=1.2.3</c>; all must hold).
/// <c>*</c> or an empty text allows every version. Prerelease versions satisfy a range only when its bounds are prereleases or
/// the caller allows them. Immutable and thread safe.
/// </summary>
public sealed class VersionRange
{
    private VersionRange(string text, SemanticVersion? minimum, bool minimumInclusive, SemanticVersion? maximum, bool maximumInclusive)
    {
        Text = text;
        Minimum = minimum;
        MinimumInclusive = minimumInclusive;
        Maximum = maximum;
        MaximumInclusive = maximumInclusive;
    }

    /// <summary>A range that every version satisfies.</summary>
    public static VersionRange All { get; } = new("*", null, false, null, false);

    /// <summary>The text the range was parsed from.</summary>
    public string Text { get; }

    /// <summary>Lower bound, or <see langword="null"/> for none.</summary>
    public SemanticVersion? Minimum { get; }

    /// <summary>Whether <see cref="Minimum"/> itself is in the range.</summary>
    public bool MinimumInclusive { get; }

    /// <summary>Upper bound, or <see langword="null"/> for none.</summary>
    public SemanticVersion? Maximum { get; }

    /// <summary>Whether <see cref="Maximum"/> itself is in the range.</summary>
    public bool MaximumInclusive { get; }

    /// <summary>A range containing exactly one version.</summary>
    /// <param name="version">The version.</param>
    /// <returns>The range.</returns>
    public static VersionRange Exactly(SemanticVersion version)
    {
        ArgumentNullException.ThrowIfNull(version);
        return new VersionRange($"[{version}]", version, true, version, true);
    }

    /// <summary>Parses a range.</summary>
    /// <param name="text">The range text.</param>
    /// <returns>The range.</returns>
    /// <exception cref="FormatException">The text is not a version range.</exception>
    public static VersionRange Parse(string? text)
    {
        return TryParse(text, out var range) ? range : throw new FormatException($"'{text}' is not a version range.");
    }

    /// <summary>Parses a range.</summary>
    /// <param name="text">The range text.</param>
    /// <param name="range">The range, when the text is valid.</param>
    /// <returns><see langword="true"/> when the text is a version range.</returns>
    public static bool TryParse(string? text, [NotNullWhen(true)] out VersionRange? range)
    {
        range = null;
        var trimmed = text?.Trim() ?? "";
        if (trimmed.Length == 0 || trimmed == "*")
        {
            range = All;
            return true;
        }

        if (trimmed[0] is '[' or '(')
        {
            return TryParseInterval(trimmed, out range);
        }

        if (char.IsAsciiDigit(trimmed[0]))
        {
            if (!SemanticVersion.TryParse(trimmed, out var floor))
            {
                return false;
            }

            range = new VersionRange(trimmed, floor, true, null, false);
            return true;
        }

        return TryParseComparators(trimmed, out range);
    }

    /// <summary>Whether <paramref name="version"/> is in the range.</summary>
    /// <param name="version">The version.</param>
    /// <param name="includePrerelease">Whether prerelease versions are allowed even when no bound is a prerelease.</param>
    /// <returns><see langword="true"/> when the version satisfies the range.</returns>
    public bool Satisfies(SemanticVersion version, bool includePrerelease = false)
    {
        ArgumentNullException.ThrowIfNull(version);
        if (version.IsPrerelease && !includePrerelease && Minimum?.IsPrerelease != true && Maximum?.IsPrerelease != true)
        {
            return false;
        }

        if (Minimum is not null)
        {
            var compared = version.CompareTo(Minimum);
            if (compared < 0 || (compared == 0 && !MinimumInclusive))
            {
                return false;
            }
        }

        if (Maximum is not null)
        {
            var compared = version.CompareTo(Maximum);
            if (compared > 0 || (compared == 0 && !MaximumInclusive))
            {
                return false;
            }
        }

        return true;
    }

    /// <inheritdoc />
    public override string ToString() => Text;

    private static bool TryParseInterval(string text, out VersionRange? range)
    {
        range = null;
        var close = text[^1];
        if (close is not (']' or ')') || text.Length < 3)
        {
            return false;
        }

        var minimumInclusive = text[0] == '[';
        var maximumInclusive = close == ']';
        var body = text[1..^1];
        var comma = body.IndexOf(',', StringComparison.Ordinal);
        if (comma < 0)
        {
            if (!minimumInclusive || !maximumInclusive || !SemanticVersion.TryParse(body, out var exact))
            {
                return false;
            }

            range = new VersionRange(text, exact, true, exact, true);
            return true;
        }

        var lower = body[..comma].Trim();
        var upper = body[(comma + 1)..].Trim();
        SemanticVersion? minimum = null;
        SemanticVersion? maximum = null;
        if (lower.Length > 0 && !SemanticVersion.TryParse(lower, out minimum))
        {
            return false;
        }

        if (upper.Length > 0 && !SemanticVersion.TryParse(upper, out maximum))
        {
            return false;
        }

        if ((minimum is null && maximum is null) || (minimum is not null && maximum is not null && minimum > maximum))
        {
            return false;
        }

        range = new VersionRange(text, minimum, minimum is not null && minimumInclusive, maximum, maximum is not null && maximumInclusive);
        return true;
    }

    private static bool TryParseComparators(string text, out VersionRange? range)
    {
        range = null;
        SemanticVersion? minimum = null;
        var minimumInclusive = false;
        SemanticVersion? maximum = null;
        var maximumInclusive = false;
        foreach (var token in text.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            var length = token.StartsWith(">=", StringComparison.Ordinal) || token.StartsWith("<=", StringComparison.Ordinal) ? 2
                : token[0] is '>' or '<' or '=' ? 1
                : 0;
            if (length == 0 || !SemanticVersion.TryParse(token[length..], out var version))
            {
                return false;
            }

            var op = token[..length];
            if (op is ">=" or ">" or "=")
            {
                var inclusive = op != ">";
                if (minimum is null || version > minimum || (version == minimum && !inclusive))
                {
                    minimum = version;
                    minimumInclusive = inclusive;
                }
            }

            if (op is "<=" or "<" or "=")
            {
                var inclusive = op != "<";
                if (maximum is null || version < maximum || (version == maximum && !inclusive))
                {
                    maximum = version;
                    maximumInclusive = inclusive;
                }
            }
        }

        range = new VersionRange(text, minimum, minimumInclusive, maximum, maximumInclusive);
        return true;
    }
}
