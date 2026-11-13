namespace Sharplight.Serve;

/// <summary>The JSON configuration file given with <c>--config</c>; command-line options take precedence over its host-wide values.</summary>
internal sealed class ServerConfig
{
    /// <summary>Device memory kept free for other applications (<c>"2G"</c>, <c>"512M"</c>, ...).</summary>
    public string? VramReserve { get; set; }

    /// <summary>Upper bound on the device memory used in total.</summary>
    public string? VramBudget { get; set; }

    /// <summary>At most this many models resident at once.</summary>
    public int? MaxLoaded { get; set; }

    /// <summary>Load models on their first request.</summary>
    public bool? Lazy { get; set; }

    /// <summary>Unload models idle for this many seconds.</summary>
    public double? IdleTimeout { get; set; }

    /// <summary>The served models.</summary>
    public List<ModelSpec>? Models { get; set; }
}
