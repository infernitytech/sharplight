namespace Sharplight.Onnx.Conformance;

/// <summary>
/// Locates the ONNX backend node tests: <c>SHARPLIGHT_ONNX_NODE_TESTS</c>, else the newest <c>~/models/onnx/backend-node-*</c>
/// directory written by <c>tools/gen-onnx-node-tests.py</c>.
/// </summary>
internal static class NodeTestSuite
{
    private static readonly Lazy<string?> s_root = new(FindRoot);

    public static string? Root => s_root.Value;

    /// <summary>Names of all node tests (directories with a model.onnx), sorted.</summary>
    public static IReadOnlyList<string> Names()
        => Root is null ? [] : Directory.GetDirectories(Root)
            .Where(d => File.Exists(Path.Combine(d, "model.onnx")))
            .Select(d => Path.GetFileName(d))
            .OrderBy(n => n, StringComparer.Ordinal)
            .ToList();

    public static string DirectoryOf(string name) => Path.Combine(Root ?? throw new InvalidOperationException("No node test directory."), name);

    private static string? FindRoot()
    {
        if (Environment.GetEnvironmentVariable("SHARPLIGHT_ONNX_NODE_TESTS") is { Length: > 0 } configured)
        {
            return Directory.Exists(configured) ? configured : null;
        }

        string models = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models", "onnx");
        if (!Directory.Exists(models))
        {
            return null;
        }

        return Directory.GetDirectories(models, "backend-node-*")
            .OrderByDescending(d => Version.TryParse(Path.GetFileName(d)["backend-node-".Length..], out var v) ? v : new Version())
            .FirstOrDefault();
    }
}
