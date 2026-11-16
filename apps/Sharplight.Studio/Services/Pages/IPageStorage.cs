namespace Sharplight.Studio.Services.Pages;

/// <summary>Where each inference page keeps what it persists: parameter presets, the run history and the output files.</summary>
internal interface IPageStorage
{
    /// <summary>The directory of a page (created when missing).</summary>
    /// <param name="pageId">The page's stable id (for example <c>transcribe</c>).</param>
    public string DirectoryFor(string pageId);

    /// <summary>The folder of the generated media of one kind (<c>images</c>), created when missing; shared by every page of that kind.</summary>
    /// <param name="kind">The kind's folder name.</param>
    public string GenerationsDirectory(string kind);
}
