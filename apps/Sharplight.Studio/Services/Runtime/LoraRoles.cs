using System.Globalization;
using Sharplight.Models;
using Sharplight.Vision;

namespace Sharplight.Studio.Services.Runtime;

/// <summary>Converts between LoRA adapters and the auxiliary roles the engine loads them by (<c>lora</c>, <c>lora:2@0.8</c>, ...).</summary>
internal static class LoraRoles
{
    /// <summary>The adapters among the auxiliary files, in role order.</summary>
    /// <param name="auxiliary">Auxiliary files by role.</param>
    public static IReadOnlyList<ModelLora> Adapters(IEnumerable<(string Role, string Path)> auxiliary)
        => [.. auxiliary.Where(entry => WeightPatches.IsPatchRole(entry.Role)).OrderBy(entry => Position(entry.Role))
            .Select(entry => new ModelLora(entry.Path, WeightPatches.Scale(entry.Role)))];

    /// <summary>The slot number of an adapter role: 1 for <c>lora</c> and <c>lora@s</c>, N for <c>lora:N</c>.</summary>
    private static int Position(string role)
    {
        int colon = role.IndexOf(':', StringComparison.Ordinal);
        if (colon < 0)
        {
            return 1;
        }

        int end = role.IndexOf('@', colon, StringComparison.Ordinal);
        return int.TryParse(role.AsSpan(colon + 1, (end < 0 ? role.Length : end) - colon - 1), NumberStyles.None, CultureInfo.InvariantCulture, out int number) ? number : int.MaxValue;
    }

    /// <summary>The auxiliary files that are not adapters.</summary>
    /// <param name="auxiliary">Auxiliary files by role.</param>
    public static IReadOnlyList<(string Role, string Path)> WithoutAdapters(IEnumerable<(string Role, string Path)> auxiliary)
        => [.. auxiliary.Where(entry => !WeightPatches.IsPatchRole(entry.Role))];

    /// <summary>The auxiliary entries of the adapters: <c>lora@s</c>, <c>lora:2@s</c>, ...</summary>
    /// <param name="adapters">The adapters.</param>
    public static IReadOnlyList<(string Role, string Path)> ToAuxiliary(IEnumerable<ModelLora> adapters)
    {
        var entries = new List<(string, string)>();
        foreach (var adapter in adapters)
        {
            string role = entries.Count == 0 ? AuxiliaryRoles.Lora : $"{AuxiliaryRoles.Lora}:{entries.Count + 1}";
            entries.Add(($"{role}@{adapter.Scale.ToString("0.###", CultureInfo.InvariantCulture)}", adapter.Path));
        }

        return entries;
    }
}
