using System.Xml.Linq;

namespace Sharplight.Studio.Tests;

/// <summary>The published Studio package leaves out the API documentation files; the NuGet packages of the libraries keep them.</summary>
public sealed class PackagingTests
{
    private static string RepositoryFile(params string[] parts)
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null && !File.Exists(Path.Combine(directory.FullName, "Sharplight.slnx")))
        {
            directory = directory.Parent;
        }

        Assert.NotNull(directory);
        return Path.Combine([directory.FullName, .. parts]);
    }

    private static string? Property(string file, string name)
        => XDocument.Load(file).Descendants(name).Select(element => element.Value).LastOrDefault();

    [Fact]
    public void Studio_DoesNotPublishXmlDocumentation_ButLibrariesStillGenerateIt()
    {
        string project = RepositoryFile("apps", "Sharplight.Studio", "Sharplight.Studio.csproj");
        Assert.Equal("false", Property(project, "PublishDocumentationFile"));
        Assert.Equal("false", Property(project, "PublishReferencesDocumentationFiles"));
        Assert.Equal("true", Property(RepositoryFile("Directory.Build.props"), "GenerateDocumentationFile"));
        // The libraries that become NuGet packages do not opt out of the xml files the packages carry.
        foreach (string library in new[] { "Sharplight.Core", "Sharplight", "Sharplight.Hosting" })
        {
            Assert.NotEqual("false", Property(RepositoryFile("src", library, library + ".csproj"), "GenerateDocumentationFile"));
        }
    }
}
