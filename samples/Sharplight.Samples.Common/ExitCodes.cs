namespace Sharplight.Samples;

/// <summary>Process exit codes shared by the samples.</summary>
public static class ExitCodes
{
    /// <summary>The command completed.</summary>
    public const int Success = 0;

    /// <summary>The command line was invalid.</summary>
    public const int UsageError = 1;

    /// <summary>The model, device or runtime failed (file not loadable, device unavailable, out of memory, ...).</summary>
    public const int RuntimeError = 2;

    /// <summary>The input data did not match what the model expects.</summary>
    public const int InvalidInput = 3;

    /// <summary>The run was cancelled with Ctrl+C (128 + SIGINT, the shell convention).</summary>
    public const int Cancelled = 130;
}
