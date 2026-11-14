namespace Sharplight.Studio.Services.Hub;

/// <summary>Progress of a pull.</summary>
/// <param name="Phase">The current step.</param>
/// <param name="Message">The latest log line.</param>
/// <param name="Received">Bytes downloaded so far (including resumed parts).</param>
/// <param name="Total">Bytes to download.</param>
/// <param name="File">The file being downloaded.</param>
/// <param name="FileReceived">Bytes of <paramref name="File"/> downloaded so far.</param>
/// <param name="FileSize">Size of <paramref name="File"/>.</param>
/// <param name="StageCompleted">Work done in a converting or compiling step (e.g. kernels compiled); 0 when unknown.</param>
/// <param name="StageTotal">Work of that step; 0 when unknown.</param>
internal sealed record PullUpdate(PullPhase Phase, string Message, long Received = 0, long Total = 0, string? File = null, long FileReceived = 0,
    long FileSize = 0, long StageCompleted = 0, long StageTotal = 0);
