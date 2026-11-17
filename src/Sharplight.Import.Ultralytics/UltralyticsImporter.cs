using System.Text.RegularExpressions;
using Sharplight.Import.Onnx;
using Sharplight.Models;

namespace Sharplight.Import.Ultralytics;

/// <summary>
/// Ultralytics YOLO importer: opens the ONNX export of a YOLOv8 / YOLO11 / YOLOv10 detection or instance segmentation model (recognized by the initializer names of the Ultralytics
/// network, <c>model.N.cv2.0.0.conv.weight</c> and so on) as the canonical architecture <c>ultralytics_yolo</c>. The class names come from the <c>names</c> entry of the export's metadata
/// (the 80 COCO names when it is missing and the model has 80 classes).
/// </summary>
/// <remarks>
/// The original <c>.pt</c> checkpoints pickle the Ultralytics classes and are not read (loading them would execute code); export them with <c>yolo export format=onnx</c>.
/// The Ultralytics weights and the export tooling are licensed AGPL-3.0 by Ultralytics (an enterprise license is available): check the license of the checkpoint before
/// shipping it. Register with <c>services.AddUltralyticsImporter()</c>.
/// </remarks>
public sealed partial class UltralyticsImporter : IModelImporter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "ultralytics_yolo";

    /// <summary>Metadata key of the head kind (<c>detect</c> or <c>segment</c>) when the export names it.</summary>
    public const string TaskKey = "ultralytics.task";

    private const long MaxFileLength = 1L << 30;

    /// <inheritdoc/>
    public string Format => "ultralytics-onnx";

    /// <inheritdoc/>
    /// <remarks>Ranks above the generic ONNX importer, which also opens the file.</remarks>
    public int Priority => 100;

    /// <inheritdoc/>
    public bool CanOpen(string path)
    {
        if (!File.Exists(path) || !path.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase) || new FileInfo(path).Length > MaxFileLength)
        {
            return false;
        }

        try
        {
            var source = OnnxImporter.Open(path);
            try
            {
                return IsUltralytics(source.Weights.Keys);
            }
            finally
            {
                Dispose(source);
            }
        }
        catch (Exception ex) when (ex is InvalidDataException or NotSupportedException or IOException)
        {
            return false;
        }
    }

    /// <inheritdoc/>
    ModelSource IModelImporter.Open(string path) => Open(path);

    /// <summary>Opens an Ultralytics ONNX export.</summary>
    /// <param name="path">The file.</param>
    /// <exception cref="InvalidDataException">The file is not an Ultralytics export.</exception>
    public static ModelSource Open(string path)
    {
        var onnx = OnnxImporter.Open(path);
        try
        {
            if (!IsUltralytics(onnx.Weights.Keys))
            {
                throw new InvalidDataException($"{path}: not an Ultralytics YOLO export.");
            }

            var metadata = new Dictionary<string, object>(onnx.Metadata, StringComparer.Ordinal)
            {
                [CanonicalKeys.Architecture] = Architecture,
                [CanonicalKeys.Name] = Path.GetFileNameWithoutExtension(path),
            };
            if (onnx.Metadata.TryGetValue("onnx.meta.names", out var names) && ParseNames(names as string ?? "") is { Length: > 0 } parsed)
            {
                metadata[CanonicalKeys.ClassifierLabels] = parsed;
            }

            if (onnx.Metadata.TryGetValue("onnx.meta.task", out var task))
            {
                metadata[TaskKey] = task;
            }

            return new ModelSource
            {
                Format = "ultralytics-onnx",
                Architecture = Architecture,
                Metadata = metadata,
                Weights = onnx.Weights,
                OwnedResources = onnx.OwnedResources,
                FormatData = onnx.FormatData,
                Path = path,
            };
        }
        catch
        {
            Dispose(onnx);
            throw;
        }
    }

    /// <summary>Parses the Python dictionary text of the <c>names</c> metadata (<c>{0: 'person', 1: 'bicycle'}</c>) into the class names by index.</summary>
    /// <param name="text">The metadata value.</param>
    /// <returns>The names, empty when the text holds none.</returns>
    internal static string[] ParseNames(string text)
    {
        var found = new SortedDictionary<int, string>();
        foreach (Match m in NameEntry().Matches(text))
        {
            string quoted = m.Groups[2].Success ? m.Groups[2].Value : m.Groups[3].Value;
            found[int.Parse(m.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture)] = quoted.Replace("\\'", "'", StringComparison.Ordinal);
        }

        if (found.Count == 0)
        {
            return [];
        }

        var names = new string[found.Keys.Max() + 1];
        for (int i = 0; i < names.Length; i++)
        {
            names[i] = found.TryGetValue(i, out var name) ? name : $"class {i}";
        }

        return names;
    }

    private static bool IsUltralytics(IEnumerable<string> weights)
    {
        bool first = false;
        bool head = false;
        foreach (string name in weights)
        {
            first |= name == "model.0.conv.weight";
            head |= HeadWeight().IsMatch(name);
        }

        return first && head;
    }

    private static void Dispose(ModelSource source)
    {
        foreach (var resource in source.OwnedResources)
        {
            resource.Dispose();
        }
    }

    [GeneratedRegex(@"(\d+)\s*:\s*(?:'((?:[^'\\]|\\.)*)'|""((?:[^""\\]|\\.)*)"")")]
    private static partial Regex NameEntry();

    [GeneratedRegex(@"^model\.\d+\.(?:cv3|one2one_cv3)\.0\.(?:0\.)?(?:0\.)?conv\.weight$")]
    private static partial Regex HeadWeight();
}
