using Sharplight.Distributed.Transport;

namespace Sharplight.Distributed;

/// <summary>Settings of the remote execution provider (<see cref="RemoteExecutionProvider"/>): the workers and the failure semantics.</summary>
public sealed class RemoteSegmentOptions
{
    /// <summary>The segment workers in pipeline order, as <c>host:port</c>.</summary>
    public IList<string> Workers { get; set; } = [];

    /// <summary>How the coordinator reaches the workers. Default: <see cref="TcpSegmentTransport"/> without TLS.</summary>
    public ISegmentTransport? Transport { get; set; }

    /// <summary>Heartbeat period on an idle connection. Default 1 s.</summary>
    public TimeSpan HeartbeatInterval { get; set; } = TimeSpan.FromSeconds(1);

    /// <summary>A worker silent for this long (no reply, no heartbeat) is considered dead and its requests fail. Default 5 s.</summary>
    public TimeSpan HeartbeatTimeout { get; set; } = TimeSpan.FromSeconds(5);

    /// <summary>Longest a single execution (one forward pass of a segment) may take before it fails. Default 10 min.</summary>
    public TimeSpan ExecuteTimeout { get; set; } = TimeSpan.FromMinutes(10);

    /// <summary>Longest a segment compilation (reading its weights, uploading them, compiling kernels) may take. Default 60 min.</summary>
    public TimeSpan CompileTimeout { get; set; } = TimeSpan.FromMinutes(60);

    /// <summary>How workers check their model copy against the coordinator's. Default <see cref="WeightVerification.Sampled"/>.</summary>
    public WeightVerification WeightVerification { get; set; } = WeightVerification.Sampled;

    /// <summary>
    /// Whether a boundary value between two remote segments goes straight from one worker to the next (default) instead of through the
    /// coordinator. <c>SHARPLIGHT_REMOTE_FORWARDING=0</c> also turns it off.
    /// </summary>
    public bool PeerForwarding { get; set; } = Environment.GetEnvironmentVariable("SHARPLIGHT_REMOTE_FORWARDING") is not ("0" or "false");

    /// <summary>The transport to use: <see cref="Transport"/>, or TCP without TLS.</summary>
    internal ISegmentTransport ResolvedTransport => Transport ?? new TcpSegmentTransport();
}
