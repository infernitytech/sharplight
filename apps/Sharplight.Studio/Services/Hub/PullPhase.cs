namespace Sharplight.Studio.Services.Hub;

/// <summary>The step a pull is in.</summary>
internal enum PullPhase
{
    /// <summary>Waiting to start.</summary>
    Queued,

    /// <summary>Resolving the revision and choosing files.</summary>
    Resolving,

    /// <summary>Downloading files.</summary>
    Downloading,

    /// <summary>Converting to a <c>.slm</c> package.</summary>
    Converting,

    /// <summary>Compiling GPU kernels into the package.</summary>
    Precompiling,

    /// <summary>Stored in the library.</summary>
    Completed,

    /// <summary>Stopped by the user; the downloaded parts are kept.</summary>
    Paused,

    /// <summary>Stopped by the user; the downloaded parts were deleted.</summary>
    Cancelled,

    /// <summary>Failed (see the message).</summary>
    Failed,
}
