namespace Sharplight.Studio.Services.Search;

/// <summary>
/// Decides how to open a document at the place a result points to. Only programs known to take a position on their command line are
/// used for that (VS Code for lines, Okular and Evince for PDF pages); anything else opens the document in the system's default program.
/// </summary>
internal static class LocationOpener
{
    /// <summary>The program and arguments that open <paramref name="path"/> at the page or line, or null for the plain default-app open.</summary>
    /// <param name="path">The document.</param>
    /// <param name="page">The page, or 0.</param>
    /// <param name="line">The line, or 0.</param>
    /// <param name="find">Finds a program on the search path (its full path), or null when it is not installed.</param>
    public static (string Program, string[] Arguments)? Command(string path, int page, int line, Func<string, string?> find)
    {
        string extension = Path.GetExtension(path).ToLowerInvariant();
        if (extension == ".pdf" && page > 0)
        {
            if (find("okular") is { } okular)
            {
                return (okular, ["--page", page.ToString(System.Globalization.CultureInfo.InvariantCulture), path]);
            }

            if (find("evince") is { } evince)
            {
                return (evince, ["--page-index=" + page.ToString(System.Globalization.CultureInfo.InvariantCulture), path]);
            }

            return null;
        }

        if (line > 0 && extension is not (".pdf" or ".docx" or ".pptx" or ".xlsx") && find("code") is { } code)
        {
            return (code, ["--goto", path + ":" + line.ToString(System.Globalization.CultureInfo.InvariantCulture)]);
        }

        return null;
    }

    /// <summary>The full path of <paramref name="program"/> on the PATH, or null.</summary>
    /// <param name="program">The program's name without extension.</param>
    public static string? FindOnPath(string program)
    {
        foreach (string directory in (Environment.GetEnvironmentVariable("PATH") ?? "").Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries))
        {
            foreach (string name in OperatingSystem.IsWindows() ? new[] { program + ".exe", program + ".cmd" } : new[] { program })
            {
                string candidate = Path.Combine(directory, name);
                if (File.Exists(candidate))
                {
                    return candidate;
                }
            }
        }

        return null;
    }
}
