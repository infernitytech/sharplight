namespace Sharplight.Onnx.Conformance;

/// <summary>
/// Locates the extra node-format tests (see <see cref="ExtraConformanceTests"/>): the directories of <c>SHARPLIGHT_ONNX_EXTRA_TESTS</c>,
/// else every <c>~/models/onnx/sharplight-*-tests</c> directory. Test names are the directory names; the generators keep them unique.
/// </summary>
internal static class ExtraTestSuite
{
    private static readonly Lazy<Dictionary<string, string>> s_tests = new(FindTests);

    /// <summary>Names of all extra tests, sorted.</summary>
    public static IReadOnlyList<string> Names() => [.. s_tests.Value.Keys.Order(StringComparer.Ordinal)];

    public static string DirectoryOf(string name) => s_tests.Value[name];

    private static Dictionary<string, string> FindTests()
    {
        var tests = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var root in Roots().Where(Directory.Exists))
        {
            foreach (var directory in Directory.GetDirectories(root).Where(d => File.Exists(Path.Combine(d, "model.onnx"))))
            {
                tests.TryAdd(Path.GetFileName(directory), directory);
            }
        }

        return tests;
    }

    private static IEnumerable<string> Roots()
    {
        if (Environment.GetEnvironmentVariable("SHARPLIGHT_ONNX_EXTRA_TESTS") is { Length: > 0 } configured)
        {
            return configured.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries);
        }

        string models = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models", "onnx");
        return Directory.Exists(models) ? Directory.GetDirectories(models, "sharplight-*-tests").Order(StringComparer.Ordinal) : [];
    }
}
