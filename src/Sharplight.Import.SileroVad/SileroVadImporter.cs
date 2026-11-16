using Sharplight.Import.Onnx;
using Sharplight.Models;

namespace Sharplight.Import.SileroVad;

/// <summary>
/// Silero VAD importer: maps the 16 kHz ONNX export of Silero VAD v5 (<c>silero_vad_16k_op15.onnx</c>: the variant without the sample-rate branches; its initializers are the
/// learned STFT basis, four convolution blocks, one LSTM cell and the decoder convolution) onto the canonical <see cref="ModelSource"/> of architecture <c>silero_vad</c>.
/// </summary>
/// <remarks>Register with <c>services.AddSileroVadImporter()</c> so that the <see cref="ModelLoader"/> opens Silero VAD files, or call <see cref="Open"/>.</remarks>
public sealed class SileroVadImporter : IModelImporter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "silero_vad";

    private const string Basis = "model.stft.forward_basis_buffer";

    private static readonly (string From, string To)[] s_names =
    [
        (Basis, "stft.basis"),
        ("model.decoder.rnn.weight_ih", "rnn.weight_ih"),
        ("model.decoder.rnn.weight_hh", "rnn.weight_hh"),
        ("model.decoder.rnn.bias_ih", "rnn.bias_ih"),
        ("model.decoder.rnn.bias_hh", "rnn.bias_hh"),
        ("model.decoder.decoder.2.weight", "head.weight"),
        ("model.decoder.decoder.2.bias", "head.bias"),
    ];

    /// <inheritdoc/>
    public string Format => "silero-vad";

    /// <inheritdoc/>
    /// <remarks>Ranks above the generic ONNX importer, which also opens the file.</remarks>
    public int Priority => 100;

    /// <inheritdoc/>
    public bool CanOpen(string path)
    {
        if (!File.Exists(path) || !path.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase) || new FileInfo(path).Length > 64 * 1024 * 1024)
        {
            return false;
        }

        try
        {
            var source = OnnxImporter.Open(path);
            try
            {
                return source.Weights.ContainsKey(Basis) && source.Weights.ContainsKey("model.encoder.0.reparam_conv.weight");
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

    /// <summary>Opens a Silero VAD ONNX file.</summary>
    /// <param name="path">The file.</param>
    /// <exception cref="InvalidDataException">The file is not a Silero VAD export.</exception>
    public static ModelSource Open(string path)
    {
        var onnx = OnnxImporter.Open(path);
        try
        {
            var weights = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
            foreach (var (from, to) in s_names)
            {
                weights[to] = (onnx.Weights.TryGetValue(from, out var w) ? w : throw new InvalidDataException($"{path}: not a Silero VAD export (no {from}).")).As(to);
            }

            int encoders = 0;
            while (onnx.Weights.ContainsKey($"model.encoder.{encoders}.reparam_conv.weight"))
            {
                weights[$"encoder.{encoders}.weight"] = onnx.Weights[$"model.encoder.{encoders}.reparam_conv.weight"].As($"encoder.{encoders}.weight");
                weights[$"encoder.{encoders}.bias"] = onnx.Weights[$"model.encoder.{encoders}.reparam_conv.bias"].As($"encoder.{encoders}.bias");
                encoders++;
            }

            long nFft = weights["stft.basis"].Shape[2].Fixed;
            string a = Architecture + ".";
            var metadata = new Dictionary<string, object>(StringComparer.Ordinal)
            {
                [CanonicalKeys.Architecture] = Architecture,
                [CanonicalKeys.Name] = Path.GetFileNameWithoutExtension(path),
                [a + "sample_rate"] = 16000,
                [a + "frame_samples"] = 512,
                [a + "context_samples"] = 64,
                [a + "hop"] = (int)(nFft / 2),
                [a + "pad"] = 64,
                [a + "hidden_size"] = (int)weights["rnn.weight_hh"].Shape[1].Fixed,
                [a + "strides"] = new long[] { 1, 2, 2, 1 },
            };
            return new ModelSource
            {
                Format = "silero-vad",
                Architecture = Architecture,
                Metadata = metadata,
                Weights = weights,
                OwnedResources = onnx.OwnedResources,
                Path = path,
            };
        }
        catch
        {
            Dispose(onnx);
            throw;
        }
    }

    private static void Dispose(ModelSource source)
    {
        foreach (var resource in source.OwnedResources)
        {
            resource.Dispose();
        }
    }
}
