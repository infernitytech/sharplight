using System.Text.RegularExpressions;
using Xunit;

namespace Sharplight.Core.Tests.Storage;

/// <summary>
/// Every component gets its data paths from the storage layout: no source file of the libraries, Studio or the samples asks the
/// operating system for well-known folders, the temporary folder or the application folder, except the few listed here that only
/// read from them (native libraries next to the app, the web content root) and the layout's own environment.
/// </summary>
public sealed partial class StorageWritersTests
{
    /// <summary>Files allowed to use the patterns, with the reason.</summary>
    private static readonly Dictionary<string, string> s_allowed = new(StringComparer.Ordinal)
    {
        ["src/Sharplight.Import.Diffusers/T5Vocabulary.cs"] = "Not a data path: a directory name that cannot exist, so the tokenizer translator finds no companion files for the embedded T5 vocabulary.",
        ["src/Sharplight.Core/Storage/StorageEnvironment.cs"] = "the one place that reads the process environment for the layout",
        ["src/Sharplight.Core/Storage/SystemStorageFolders.cs"] = "the platform conventions of the system folders",
        ["src/Sharplight.Audio/Native/NativeAudioLibrary.cs"] = "loads the native library shipped next to the app (read only)",
        ["src/Sharplight.Video/Native/NativeVideoLibrary.cs"] = "loads the native video library shipped next to the app (read only)",
        ["src/Sharplight.Vulkan/VulkanLibrary.cs"] = "loads the Vulkan loader shipped next to the app (read only)",
        ["apps/Sharplight.Studio/Services/Server/KestrelApiServer.cs"] = "ASP.NET content root (read only, no static files)",
        ["samples/Sharplight.Server/HttpServer.cs"] = "ASP.NET content root (read only, no static files)",
    };

    [Fact]
    public void NoComponent_PicksItsOwnDataFolder()
    {
        var root = FindRepositoryRoot();
        var offenders = new List<string>();
        foreach (var folder in new[] { "src", "apps", "samples" })
        {
            foreach (var file in Directory.EnumerateFiles(Path.Combine(root, folder), "*.cs", SearchOption.AllDirectories))
            {
                var relative = Path.GetRelativePath(root, file).Replace('\\', '/');
                if (relative.Contains("/obj/", StringComparison.Ordinal) || relative.Contains("/bin/", StringComparison.Ordinal) || s_allowed.ContainsKey(relative))
                {
                    continue;
                }

                var lines = File.ReadAllLines(file);
                for (int i = 0; i < lines.Length; i++)
                {
                    var code = lines[i].TrimStart();
                    if (!code.StartsWith("//", StringComparison.Ordinal) && Forbidden().IsMatch(code))
                    {
                        offenders.Add($"{relative}:{i + 1}: {code}");
                    }
                }
            }
        }

        Assert.True(offenders.Count == 0, "Use StorageLayout (Sharplight.Storage) for data paths instead of:\n" + string.Join("\n", offenders));
        Assert.All(s_allowed.Keys, path => Assert.True(File.Exists(Path.Combine(root, path)), $"{path} is allowed but no longer exists."));
    }

    private static string FindRepositoryRoot()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null && !File.Exists(Path.Combine(directory.FullName, "Sharplight.slnx")))
        {
            directory = directory.Parent;
        }

        return directory?.FullName ?? throw new InvalidOperationException("The repository root (Sharplight.slnx) was not found.");
    }

    [GeneratedRegex(@"GetFolderPath\(|SpecialFolder\.|GetTempPath\(|GetTempFileName\(|AppContext\.BaseDirectory|""XDG_[A-Z_]+""|""(LOCAL)?APPDATA""")]
    private static partial Regex Forbidden();
}
