namespace Sharplight.Vulkan;

/// <summary>
/// Locates the Vulkan library to load. On macOS there is no system Vulkan loader, so a MoltenVK library shipped with the
/// application (the <c>Sharplight.Vulkan.MoltenVK</c> package, or <c>Contents/Frameworks</c> of an app bundle) is
/// preferred; <c>SHARPLIGHT_VULKAN_LIBRARY</c> names another library (e.g. the Vulkan SDK's <c>libvulkan.1.dylib</c>).
/// Other platforms always use the system loader.
/// </summary>
internal static class VulkanLibrary
{
    /// <summary>Environment variable naming the Vulkan library to load on macOS.</summary>
    public const string EnvironmentVariable = "SHARPLIGHT_VULKAN_LIBRARY";

    /// <summary>File name of the bundled MoltenVK library.</summary>
    public const string MoltenVKFileName = "libMoltenVK.dylib";

    /// <summary>
    /// The library to try before the default probing of Vortice.Vulkan (which falls back to <c>libvulkan.dylib</c>,
    /// <c>libMoltenVK.dylib</c> and the frameworks on the dynamic linker's search path), or null to use the defaults only.
    /// </summary>
    public static string? Resolve()
    {
        if (!OperatingSystem.IsMacOS())
        {
            return null;
        }

        string? configured = Environment.GetEnvironmentVariable(EnvironmentVariable);
        if (!string.IsNullOrWhiteSpace(configured))
        {
            return configured;
        }

        return Candidates().FirstOrDefault(File.Exists);
    }

    /// <summary>Where a bundled MoltenVK can be: next to the application, in its NuGet runtime folder, or in an app bundle.</summary>
    private static IEnumerable<string> Candidates()
    {
        string baseDirectory = AppContext.BaseDirectory;
        yield return Path.Combine(baseDirectory, MoltenVKFileName);
        yield return Path.Combine(baseDirectory, "runtimes", "osx-arm64", "native", MoltenVKFileName);
        yield return Path.GetFullPath(Path.Combine(baseDirectory, "..", "Frameworks", MoltenVKFileName));
        if (AppContext.GetData("NATIVE_DLL_SEARCH_DIRECTORIES") is string searchDirectories)
        {
            foreach (string directory in searchDirectories.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries))
            {
                yield return Path.Combine(directory, MoltenVKFileName);
            }
        }
    }
}
