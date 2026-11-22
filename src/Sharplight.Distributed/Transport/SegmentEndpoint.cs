using System.Diagnostics.CodeAnalysis;
using System.Globalization;

namespace Sharplight.Distributed.Transport;

/// <summary>The network address of a segment worker: a host name or IP address and a port (<c>host:port</c>, IPv6 as <c>[::1]:7070</c>).</summary>
/// <param name="Host">Host name or IP address.</param>
/// <param name="Port">TCP port (1-65535; 0 when listening asks the system for a free port).</param>
public sealed record SegmentEndpoint(string Host, int Port)
{
    /// <summary>Parses <c>host:port</c> or <c>[ipv6]:port</c>.</summary>
    /// <exception cref="FormatException">The text is not an endpoint.</exception>
    public static SegmentEndpoint Parse(string text)
        => TryParse(text, out var endpoint) ? endpoint : throw new FormatException($"'{text}' is not an endpoint (host:port or [ipv6]:port).");

    /// <summary>Parses <c>host:port</c> or <c>[ipv6]:port</c>.</summary>
    /// <param name="text">The text.</param>
    /// <param name="endpoint">The endpoint, when the text is one.</param>
    /// <returns>Whether the text is an endpoint.</returns>
    public static bool TryParse(string? text, [NotNullWhen(true)] out SegmentEndpoint? endpoint)
    {
        endpoint = null;
        if (string.IsNullOrWhiteSpace(text))
        {
            return false;
        }

        text = text.Trim();
        int colon = text.LastIndexOf(':');
        if (colon <= 0 || colon == text.Length - 1)
        {
            return false;
        }

        string host = text[..colon];
        if (host.StartsWith('[') && host.EndsWith(']'))
        {
            host = host[1..^1];
        }
        else if (host.Contains(':'))
        {
            return false;
        }

        if (host.Length == 0 || !int.TryParse(text.AsSpan(colon + 1), NumberStyles.None, CultureInfo.InvariantCulture, out int port) || port > 65535)
        {
            return false;
        }

        endpoint = new SegmentEndpoint(host, port);
        return true;
    }

    /// <summary>Parses a comma-separated list of endpoints (empty entries are skipped).</summary>
    /// <exception cref="FormatException">An entry is not an endpoint.</exception>
    public static IReadOnlyList<SegmentEndpoint> ParseList(string? text)
        => string.IsNullOrWhiteSpace(text) ? [] : [.. text.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(Parse)];

    /// <inheritdoc/>
    public override string ToString() => Host.Contains(':') ? $"[{Host}]:{Port.ToString(CultureInfo.InvariantCulture)}" : $"{Host}:{Port.ToString(CultureInfo.InvariantCulture)}";
}
