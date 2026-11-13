using Sharplight.Package;

namespace Sharplight.PackageViewer.ViewModels;

/// <summary>One entry of the section table.</summary>
public sealed record SectionRow(string Kind, string Name, long Offset, long Length)
{
    public string Size => ByteSizeFormat.Format(Length);

    public static SectionRow From(PackageSectionInfo section) => new(section.Kind, section.Name, section.Offset, section.Length);
}
