using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Onnx;

/// <summary>
/// ONNX importer: memory-maps the model file, decodes it with the built-in protobuf reader and exposes
/// the initializers as zero-copy weights. The parsed graph travels in <see cref="ModelSource.FormatData"/>
/// for <see cref="OnnxGraphAdapter"/>, which translates it into Sharplight's IR.
/// </summary>
public sealed class OnnxImporter : IModelImporter
{
    /// <summary>Format identifier of the importer ("onnx").</summary>
    public const string FormatName = "onnx";

    /// <summary>Architecture name of every imported ONNX model; selects <see cref="OnnxGraphAdapter"/>.</summary>
    public const string ArchitectureName = "onnx";

    /// <inheritdoc />
    public string Format => FormatName;

    /// <summary>Registers the importer and the generic ONNX graph adapter.</summary>
    public static void Register()
    {
        ModelImporterRegistry.Register(new OnnxImporter());
        ModelAdapterRegistry.Register(new OnnxGraphAdapter());
    }

    /// <summary>
    /// Cheap sniff test: true for an existing <c>.onnx</c> file whose top-level protobuf structure looks like a ModelProto.
    /// Never throws for unreadable or malformed files.
    /// </summary>
    public bool CanOpen(string path)
    {
        if (!path.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase) || !File.Exists(path))
        {
            return false;
        }

        try
        {
            return LooksLikeModelProto(path);
        }
        catch (IOException)
        {
            return false;
        }
        catch (UnauthorizedAccessException)
        {
            return false;
        }
    }

    /// <summary>Walks the top-level protobuf fields (skipping payloads) and checks for a plausible ModelProto.</summary>
    private static bool LooksLikeModelProto(string path)
    {
        using var f = File.OpenRead(path);
        long len = f.Length;
        if (len < 4)
        {
            return false;
        }

        bool sawGraph = false;
        long pos = 0;
        int fields = 0;
        var buf = new byte[10];
        while (pos < len)
        {
            f.Position = pos;
            int n = f.Read(buf);
            if (n == 0)
            {
                return false;
            }

            int i = 0;
            ulong ReadVar()
            {
                ulong v = 0;
                int sh = 0;
                while (true)
                {
                    if (i >= n || sh > 63)
                    {
                        throw new InvalidDataException();
                    }

                    byte b = buf[i++];
                    v |= (ulong)(b & 0x7F) << sh;
                    if ((b & 0x80) == 0)
                    {
                        return v;
                    }

                    sh += 7;
                }
            }
            try
            {
                ulong key = ReadVar();
                int field = (int)(key >> 3);
                int wire = (int)(key & 7);
                if (field is <= 0 or > 30)
                {
                    return false;
                }

                switch (wire)
                {
                    case 0:
                        ReadVar();
                        break;
                    case 1:
                        i += 8;
                        break;
                    case 5:
                        i += 4;
                        break;
                    case 2:
                        {
                            ulong l = ReadVar();
                            if ((long)l < 0 || pos + i + (long)l > len)
                            {
                                return false;
                            }

                            if (field == 7)
                            {
                                sawGraph = true;
                            }

                            pos += (long)l;
                            break;
                        }
                    default:
                        return false;
                }
            }
            catch (InvalidDataException)
            {
                return false;
            }
            pos += i;
            if (++fields > 64 && !sawGraph)
            {
                return false;
            }
        }
        return pos == len && sawGraph;
    }

    ModelSource IModelImporter.Open(string path) => Open(path);

    /// <summary>
    /// Memory-maps and parses an ONNX model. Initializers become zero-copy weights backed by the mapped file(s); the mappings are
    /// owned by the returned source and released when it is disposed.
    /// </summary>
    /// <param name="path">Path of the <c>.onnx</c> file; external tensor data is resolved relative to its directory.</param>
    /// <param name="options">Input shape overrides and translation options; defaults are used when null.</param>
    /// <returns>A source whose <see cref="ModelSource.FormatData"/> holds the parsed model for <see cref="OnnxGraphAdapter"/>.</returns>
    /// <exception cref="InvalidDataException">The file is not a valid ONNX protobuf or references data outside the model directory.</exception>
    /// <exception cref="NotSupportedException">The file exceeds the 2 GiB protobuf limit.</exception>
    public static ModelSource Open(string path, OnnxImportOptions? options = null)
    {
        var resources = new List<IDisposable>();
        try
        {
            var file = new MappedFile(path);
            resources.Add(file);
            if (file.Length > int.MaxValue)
            {
                throw new NotSupportedException($"{path}: protobuf files over 2 GiB are invalid; store tensors as external data.");
            }

            var ext = new ExternalDataResolver(Path.GetDirectoryName(Path.GetFullPath(path))!, resources);
            var model = OnnxParser.Parse(file.AsSpan(), new MappedSliceProvider(file), ext);
            if (options is not null)
            {
                model.Options = options;
            }

            return FromModel(model, path, resources);
        }
        catch
        {
            foreach (var r in resources)
            {
                r.Dispose();
            }

            throw;
        }
    }

    /// <summary>Wraps an already parsed model (e.g. from <see cref="OnnxParser.Parse(byte[], IOnnxExternalDataResolver?)"/>).</summary>
    internal static ModelSource FromModel(OnnxModel model, string? path = null, IReadOnlyList<IDisposable>? resources = null)
    {
        var weights = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
        foreach (var t in model.Graph.Initializers)
        {
            if (ToWeight(t) is { } w)
            {
                weights[t.Name] = w;
            }
        }

        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [CanonicalKeys.Architecture] = ArchitectureName,
            [CanonicalKeys.Name] = model.Graph.Name,
            ["onnx.ir_version"] = model.IrVersion,
            ["onnx.producer_name"] = model.ProducerName,
            ["onnx.producer_version"] = model.ProducerVersion,
            ["onnx.model_version"] = model.ModelVersion,
            ["onnx.domain"] = model.Domain,
            ["onnx.doc_string"] = model.DocString,
            ["onnx.opset"] = model.Opset(),
        };
        foreach (var (d, v) in model.Opsets)
        {
            if (d != "")
            {
                md["onnx.opset." + d] = v;
            }
        }

        foreach (var (k, v) in model.MetadataProps)
        {
            md["onnx.meta." + k] = v;
        }

        return new ModelSource
        {
            Format = FormatName,
            Architecture = ArchitectureName,
            Metadata = md,
            Weights = weights,
            OwnedResources = resources ?? [],
            FormatData = model,
            Path = path,
        };
    }

    internal static DataType? ToDataType(int onnxType) => onnxType switch
    {
        OnnxType.Float => DataType.F32,
        OnnxType.Float16 => DataType.F16,
        OnnxType.BFloat16 => DataType.BF16,
        OnnxType.Double => DataType.F64,
        OnnxType.Int8 => DataType.I8,
        OnnxType.UInt8 => DataType.U8,
        OnnxType.Int16 => DataType.I16,
        OnnxType.Int32 => DataType.I32,
        OnnxType.UInt32 => DataType.U32,
        OnnxType.Int64 => DataType.I64,
        OnnxType.Bool => DataType.Bool,
        _ => null,
    };

    private static WeightInfo? ToWeight(OnnxTensor t)
    {
        if (t.Source is null || ToDataType(t.ElemType) is not { } dt)
        {
            return null;
        }

        return new WeightInfo { Name = t.Name, Type = dt, Shape = new Shape(t.Dims.Select(d => (Dim)d).ToArray()), Source = t.Source };
    }
}
