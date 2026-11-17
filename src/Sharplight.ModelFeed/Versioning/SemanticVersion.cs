using System.Diagnostics.CodeAnalysis;
using System.Globalization;

namespace Sharplight.ModelFeed.Versioning;

/// <summary>
/// A Semantic Versioning 2.0.0 version (<c>MAJOR.MINOR.PATCH[-prerelease][+build]</c>). Comparison follows SemVer precedence: build
/// metadata is ignored, a prerelease sorts before its release, prerelease identifiers compare numerically when both are numeric.
/// Immutable and thread safe.
/// </summary>
public sealed class SemanticVersion : IComparable<SemanticVersion>, IEquatable<SemanticVersion>
{
    private SemanticVersion(long major, long minor, long patch, string prerelease, string build)
    {
        Major = major;
        Minor = minor;
        Patch = patch;
        Prerelease = prerelease;
        Build = build;
    }

    /// <summary>Major version.</summary>
    public long Major { get; }

    /// <summary>Minor version.</summary>
    public long Minor { get; }

    /// <summary>Patch version.</summary>
    public long Patch { get; }

    /// <summary>Prerelease label without the leading hyphen, or empty.</summary>
    public string Prerelease { get; }

    /// <summary>Build metadata without the leading plus, or empty (ignored for precedence).</summary>
    public string Build { get; }

    /// <summary>Whether the version has a prerelease label.</summary>
    public bool IsPrerelease => Prerelease.Length > 0;

    /// <summary>Creates a release version.</summary>
    /// <param name="major">Major version.</param>
    /// <param name="minor">Minor version.</param>
    /// <param name="patch">Patch version.</param>
    /// <returns>The version.</returns>
    public static SemanticVersion Create(long major, long minor, long patch)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(major);
        ArgumentOutOfRangeException.ThrowIfNegative(minor);
        ArgumentOutOfRangeException.ThrowIfNegative(patch);
        return new SemanticVersion(major, minor, patch, "", "");
    }

    /// <summary>Parses a version.</summary>
    /// <param name="text">The version text.</param>
    /// <returns>The version.</returns>
    /// <exception cref="FormatException">The text is not a SemVer 2 version.</exception>
    public static SemanticVersion Parse(string text)
    {
        return TryParse(text, out var version) ? version : throw new FormatException($"'{text}' is not a semantic version (MAJOR.MINOR.PATCH).");
    }

    /// <summary>Parses a version.</summary>
    /// <param name="text">The version text.</param>
    /// <param name="version">The version, when the text is valid.</param>
    /// <returns><see langword="true"/> when the text is a SemVer 2 version.</returns>
    public static bool TryParse([NotNullWhen(true)] string? text, [NotNullWhen(true)] out SemanticVersion? version)
    {
        version = null;
        if (string.IsNullOrWhiteSpace(text))
        {
            return false;
        }

        var core = text.Trim();
        var build = "";
        var plus = core.IndexOf('+', StringComparison.Ordinal);
        if (plus >= 0)
        {
            build = core[(plus + 1)..];
            core = core[..plus];
            if (!ValidIdentifiers(build, numericRule: false))
            {
                return false;
            }
        }

        var prerelease = "";
        var hyphen = core.IndexOf('-', StringComparison.Ordinal);
        if (hyphen >= 0)
        {
            prerelease = core[(hyphen + 1)..];
            core = core[..hyphen];
            if (!ValidIdentifiers(prerelease, numericRule: true))
            {
                return false;
            }
        }

        var parts = core.Split('.');
        if (parts.Length != 3 || !TryNumber(parts[0], out var major) || !TryNumber(parts[1], out var minor) || !TryNumber(parts[2], out var patch))
        {
            return false;
        }

        version = new SemanticVersion(major, minor, patch, prerelease, build);
        return true;
    }

    /// <summary>The next major version (<c>MAJOR+1.0.0</c>).</summary>
    /// <returns>The version.</returns>
    public SemanticVersion NextMajor() => Create(Major + 1, 0, 0);

    /// <summary>The next minor version (<c>MAJOR.MINOR+1.0</c>).</summary>
    /// <returns>The version.</returns>
    public SemanticVersion NextMinor() => Create(Major, Minor + 1, 0);

    /// <summary>The next patch version (<c>MAJOR.MINOR.PATCH+1</c>).</summary>
    /// <returns>The version.</returns>
    public SemanticVersion NextPatch() => Create(Major, Minor, Patch + 1);

    /// <inheritdoc />
    public int CompareTo(SemanticVersion? other)
    {
        if (other is null)
        {
            return 1;
        }

        var result = Major.CompareTo(other.Major);
        if (result == 0)
        {
            result = Minor.CompareTo(other.Minor);
        }

        if (result == 0)
        {
            result = Patch.CompareTo(other.Patch);
        }

        return result != 0 ? result : ComparePrerelease(Prerelease, other.Prerelease);
    }

    /// <inheritdoc />
    public bool Equals(SemanticVersion? other) => other is not null && CompareTo(other) == 0;

    /// <inheritdoc />
    public override bool Equals(object? obj) => obj is SemanticVersion other && Equals(other);

    /// <inheritdoc />
    public override int GetHashCode() => HashCode.Combine(Major, Minor, Patch, Prerelease);

    /// <inheritdoc />
    public override string ToString()
    {
        var text = string.Create(CultureInfo.InvariantCulture, $"{Major}.{Minor}.{Patch}");
        if (Prerelease.Length > 0)
        {
            text += "-" + Prerelease;
        }

        return Build.Length > 0 ? text + "+" + Build : text;
    }

    /// <summary>Precedence comparison.</summary>
    /// <param name="left">Left version.</param>
    /// <param name="right">Right version.</param>
    /// <returns>Whether left sorts before right.</returns>
    public static bool operator <(SemanticVersion left, SemanticVersion right) => Compare(left, right) < 0;

    /// <summary>Precedence comparison.</summary>
    /// <param name="left">Left version.</param>
    /// <param name="right">Right version.</param>
    /// <returns>Whether left sorts after right.</returns>
    public static bool operator >(SemanticVersion left, SemanticVersion right) => Compare(left, right) > 0;

    /// <summary>Precedence comparison.</summary>
    /// <param name="left">Left version.</param>
    /// <param name="right">Right version.</param>
    /// <returns>Whether left sorts before or equal to right.</returns>
    public static bool operator <=(SemanticVersion left, SemanticVersion right) => Compare(left, right) <= 0;

    /// <summary>Precedence comparison.</summary>
    /// <param name="left">Left version.</param>
    /// <param name="right">Right version.</param>
    /// <returns>Whether left sorts after or equal to right.</returns>
    public static bool operator >=(SemanticVersion left, SemanticVersion right) => Compare(left, right) >= 0;

    /// <summary>Precedence equality.</summary>
    /// <param name="left">Left version.</param>
    /// <param name="right">Right version.</param>
    /// <returns>Whether both have the same precedence.</returns>
    public static bool operator ==(SemanticVersion? left, SemanticVersion? right) => left is null ? right is null : left.Equals(right);

    /// <summary>Precedence inequality.</summary>
    /// <param name="left">Left version.</param>
    /// <param name="right">Right version.</param>
    /// <returns>Whether the precedence differs.</returns>
    public static bool operator !=(SemanticVersion? left, SemanticVersion? right) => !(left == right);

    private static int Compare(SemanticVersion left, SemanticVersion right)
    {
        ArgumentNullException.ThrowIfNull(left);
        return left.CompareTo(right);
    }

    private static int ComparePrerelease(string left, string right)
    {
        if (left.Length == 0 || right.Length == 0)
        {
            return right.Length.CompareTo(left.Length);
        }

        var leftParts = left.Split('.');
        var rightParts = right.Split('.');
        for (var index = 0; index < Math.Min(leftParts.Length, rightParts.Length); index++)
        {
            var leftNumeric = long.TryParse(leftParts[index], NumberStyles.None, CultureInfo.InvariantCulture, out var leftNumber);
            var rightNumeric = long.TryParse(rightParts[index], NumberStyles.None, CultureInfo.InvariantCulture, out var rightNumber);
            int result;
            if (leftNumeric && rightNumeric)
            {
                result = leftNumber.CompareTo(rightNumber);
            }
            else if (leftNumeric != rightNumeric)
            {
                result = leftNumeric ? -1 : 1;
            }
            else
            {
                result = string.CompareOrdinal(leftParts[index], rightParts[index]);
            }

            if (result != 0)
            {
                return result;
            }
        }

        return leftParts.Length.CompareTo(rightParts.Length);
    }

    private static bool TryNumber(string text, out long value)
    {
        value = 0;
        if (text.Length == 0 || (text.Length > 1 && text[0] == '0'))
        {
            return false;
        }

        return long.TryParse(text, NumberStyles.None, CultureInfo.InvariantCulture, out value);
    }

    private static bool ValidIdentifiers(string text, bool numericRule)
    {
        if (text.Length == 0)
        {
            return false;
        }

        foreach (var identifier in text.Split('.'))
        {
            if (identifier.Length == 0 || !identifier.All(character => char.IsAsciiLetterOrDigit(character) || character == '-'))
            {
                return false;
            }

            if (numericRule && identifier.Length > 1 && identifier[0] == '0' && identifier.All(char.IsAsciiDigit))
            {
                return false;
            }
        }

        return true;
    }
}
