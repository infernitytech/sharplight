namespace Sharplight.Studio.Services.Pages;

/// <summary>The file a page keeps its run history in (<c>history.json</c>), newest first.</summary>
internal sealed class RunHistoryFile
{
    /// <summary>The runs.</summary>
    public List<RunRecord> Runs { get; set; } = [];
}
