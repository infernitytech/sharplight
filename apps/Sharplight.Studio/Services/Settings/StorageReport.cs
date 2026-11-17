using System.Text;
using Sharplight.Storage;

namespace Sharplight.Studio.Services.Settings;

/// <summary>The resolved storage layout of the running process as text, for <c>--print-storage</c> (diagnosing packaged builds without a display).</summary>
internal static class StorageReport
{
    /// <summary>Describes what the process sees (package variables, folders) and the layout it resolves.</summary>
    /// <param name="environment">The environment of the process.</param>
    public static string Create(StorageEnvironment environment)
    {
        var layout = StorageLayout.Resolve(environment);
        var text = new StringBuilder();
        text.AppendLine($"Location:    {layout.Description}");
        text.AppendLine($"Mode:        {layout.Mode} ({layout.Source})");
        text.AppendLine($"Application: {environment.ApplicationDirectory}");
        text.AppendLine($"APPIMAGE:    {environment.GetVariable("APPIMAGE") ?? "(not set)"}");
        text.AppendLine($"APPDIR:      {environment.GetVariable("APPDIR") ?? "(not set)"}");
        text.AppendLine($"Package:     {layout.PackageDirectory}");
        text.AppendLine($"Portable:    {layout.PortableRoot} ({(StorageLayout.CanUsePortable(environment) ? "usable" : "not usable")})");
        text.AppendLine($"Marker:      {layout.PortableMarkerFile} ({(File.Exists(layout.PortableMarkerFile) ? "present" : "absent")})");
        text.AppendLine($"Config:      {layout.ConfigDirectory}");
        text.AppendLine($"Data:        {layout.DataDirectory}");
        text.AppendLine($"Models:      {layout.ModelsDirectory}");
        text.AppendLine($"Cache:       {layout.CacheDirectory}");
        text.AppendLine($"Logs:        {layout.LogsDirectory}");
        return text.ToString();
    }
}
