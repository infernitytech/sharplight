namespace Sharplight.Studio.ViewModels.Settings;

/// <summary>One kind of files on the storage card: where it is now, and where it goes with the selected location.</summary>
/// <param name="Name">What is kept there.</param>
/// <param name="Current">The folder in use.</param>
/// <param name="Next">The folder with the selected location; empty when it does not change.</param>
internal sealed record StoragePathRow(string Name, string Current, string Next)
{
    /// <summary>The folder changes with the selected location.</summary>
    public bool Changes => Next.Length > 0;
}
