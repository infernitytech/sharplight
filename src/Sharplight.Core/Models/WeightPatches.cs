using System.Globalization;
using Sharplight.Vision;

namespace Sharplight.Models;

/// <summary>
/// Merges the weight-patch auxiliary sources (<see cref="IWeightPatch"/>, roles starting with <see cref="AuxiliaryRoles.Lora"/>) into the
/// primary before an adapter builds it, so every adapter sees patched weights.
/// </summary>
public static class WeightPatches
{
    /// <summary>The strength of a patch role: the number after <c>@</c> (<c>lora@0.8</c>), else 1.</summary>
    /// <param name="role">The auxiliary role.</param>
    /// <exception cref="ArgumentException">The suffix is not a number.</exception>
    public static float Scale(string role)
    {
        ArgumentNullException.ThrowIfNull(role);
        int at = role.LastIndexOf('@');
        if (at < 0)
        {
            return 1f;
        }

        return float.TryParse(role.AsSpan(at + 1), NumberStyles.Float, CultureInfo.InvariantCulture, out float scale)
            ? scale
            : throw new ArgumentException($"Auxiliary role '{role}': the strength after '@' is not a number.", nameof(role));
    }

    /// <summary>Whether a role names a weight patch (<c>lora</c>, <c>lora:2</c>, <c>lora@0.5</c>, ...).</summary>
    /// <param name="role">The auxiliary role.</param>
    public static bool IsPatchRole(string role)
        => role.StartsWith(AuxiliaryRoles.Lora, StringComparison.Ordinal)
           && (role.Length == AuxiliaryRoles.Lora.Length || role[AuxiliaryRoles.Lora.Length] is ':' or '@' or '-');

    /// <summary>The sources with every patch source merged into the primary (in role order); the set itself when there is none.</summary>
    /// <param name="sources">The opened sources.</param>
    /// <exception cref="NotSupportedException">A patch role holds a source that is not a weight patch.</exception>
    public static ModelSourceSet Merge(ModelSourceSet sources)
    {
        ArgumentNullException.ThrowIfNull(sources);
        var patches = sources.Auxiliary.Where(pair => IsPatchRole(pair.Key)).OrderBy(pair => pair.Key, StringComparer.Ordinal).ToList();
        if (patches.Count == 0)
        {
            return sources;
        }

        var primary = sources.Primary;
        var weights = new Dictionary<string, WeightInfo>(primary.Weights, StringComparer.Ordinal);
        var resources = new List<IDisposable>(primary.OwnedResources);
        foreach (var (role, source) in patches)
        {
            if (source.FormatData is not IWeightPatch patch)
            {
                throw new NotSupportedException($"Auxiliary role '{role}': '{source.Path}' is not a LoRA or other weight patch (architecture '{source.Architecture}').");
            }

            var current = new ModelSource
            {
                Format = primary.Format,
                Architecture = primary.Architecture,
                Metadata = primary.Metadata,
                Weights = weights,
                Path = primary.Path,
            };
            foreach (var (name, weight) in patch.Apply(current, Scale(role)))
            {
                weights[name] = weight;
            }

            resources.AddRange(source.OwnedResources);
        }

        var merged = new ModelSource
        {
            Format = primary.Format,
            Architecture = primary.Architecture,
            Metadata = primary.Metadata,
            Weights = weights,
            Attachments = primary.Attachments,
            OwnedResources = resources,
            FormatData = primary.FormatData,
            Path = primary.Path,
            Definition = primary.Definition,
            CompilationCaches = primary.CompilationCaches,
        };
        return new ModelSourceSet
        {
            Primary = merged,
            Auxiliary = sources.Auxiliary.Where(pair => !IsPatchRole(pair.Key)).ToDictionary(pair => pair.Key, pair => pair.Value),
        };
    }
}
