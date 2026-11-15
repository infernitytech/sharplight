namespace Sharplight.Studio.ViewModels.Inference;

/// <summary>Where a queued run stands.</summary>
internal enum RunStatus
{
    /// <summary>Waiting for the model's current run to finish.</summary>
    Queued,

    /// <summary>Running.</summary>
    Running,

    /// <summary>Finished.</summary>
    Completed,

    /// <summary>Failed.</summary>
    Failed,

    /// <summary>Cancelled by the user.</summary>
    Cancelled,
}
