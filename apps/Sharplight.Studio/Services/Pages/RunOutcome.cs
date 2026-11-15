namespace Sharplight.Studio.Services.Pages;

/// <summary>How a run of an inference page ended.</summary>
internal enum RunOutcome
{
    /// <summary>It produced its output.</summary>
    Completed,

    /// <summary>It failed (see <see cref="RunRecord.Error"/>).</summary>
    Failed,

    /// <summary>The user cancelled it.</summary>
    Cancelled,
}
