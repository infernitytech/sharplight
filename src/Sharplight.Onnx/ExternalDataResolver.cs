using Sharplight.Models;

namespace Sharplight.Onnx;

/// <summary>
/// Resolves external tensor data files relative to the model directory and memory-maps them; each mapping is registered in
/// <paramref name="resources"/> so the importer's <see cref="ModelSource"/> owns it.
/// </summary>
internal sealed class ExternalDataResolver(string baseDir, List<IDisposable> resources) : IOnnxExternalDataResolver
{
    private readonly Dictionary<string, MappedFile> _files = new(StringComparer.Ordinal);

    public IWeightSource Open(string location, long offset, long length)
    {
        string full = Path.GetFullPath(Path.Combine(baseDir, location));
        string root = baseDir.EndsWith(Path.DirectorySeparatorChar) ? baseDir : baseDir + Path.DirectorySeparatorChar;
        if (!full.StartsWith(root, StringComparison.Ordinal))
        {
            throw new InvalidDataException($"External data location '{location}' escapes the model directory.");
        }

        if (!_files.TryGetValue(full, out var file))
        {
            if (!File.Exists(full))
            {
                throw new FileNotFoundException($"External data file '{location}' not found next to the model.", full);
            }

            file = new MappedFile(full);
            _files[full] = file;
            resources.Add(file);
        }

        if (length < 0)
        {
            length = file.Length - offset;
        }

        if (offset < 0 || offset + length > file.Length)
        {
            throw new InvalidDataException($"External data range [{offset}, {offset + length}) is outside '{location}' ({file.Length} bytes).");
        }

        return new MappedRegionSource(file, offset, length);
    }
}
