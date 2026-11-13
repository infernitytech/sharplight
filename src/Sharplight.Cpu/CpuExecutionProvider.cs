using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Cpu;

/// <summary>Execution provider running graphs on the host CPU (multithreaded, SIMD, dequantize-on-the-fly).</summary>
public sealed class CpuExecutionProvider : IExecutionProvider
{
    public const string ProviderName = "cpu";
    public string Name => ProviderName;

    public ProviderAvailability CheckAvailability() => ProviderAvailability.Available("host CPU");

    public IReadOnlyList<DeviceInfo> GetDevices() => [CpuDevice.DescribeHost()];

    public IComputeDevice OpenDevice(DeviceInfo device, DeviceOptions? options = null)
    {
        if (device.Provider != Name)
        {
            throw new ArgumentException($"Device {device} does not belong to the cpu provider.");
        }

        return new CpuDevice(device, options);
    }

    public void Dispose() { }

    /// <summary>Registers the CPU provider with <see cref="ProviderRegistry"/>.</summary>
    public static void Register() => ProviderRegistry.Register(() => new CpuExecutionProvider());
}

/// <summary>The host CPU as a compute device.</summary>
public sealed class CpuDevice : IComputeDevice
{
    private readonly DeviceOptions? _options;
    public DeviceInfo Info { get; }

    internal CpuDevice(DeviceInfo info, DeviceOptions? options)
    {
        Info = info;
        _options = options;
        if (options?.Settings.TryGetValue("threads", out var t) == true && int.TryParse(t, out var n) && n > 0)
        {
            Par.Threads = n;
        }
    }

    public long AvailableMemory
    {
        get
        {
            long avail = MemInfo("MemAvailable") ?? Info.TotalMemory;
            return _options?.MemoryBudget is { } b ? Math.Min(b, avail) : avail;
        }
    }

    internal static DeviceInfo DescribeHost()
    {
        string name = CpuName();
        var caps = new Dictionary<string, string>
        {
            ["threads"] = Environment.ProcessorCount.ToString(),
            ["arch"] = RuntimeInformation.ProcessArchitecture.ToString().ToLowerInvariant(),
            ["avx2"] = Avx2.IsSupported ? "true" : "false",
            ["avx512"] = Avx512F.IsSupported ? "true" : "false",
            ["avx512bw"] = Avx512BW.IsSupported ? "true" : "false",
            ["fma"] = Fma.IsSupported ? "true" : "false",
            ["f16c"] = Avx2.IsSupported ? "true" : "false",
            ["avx_vnni"] = AvxVnni.IsSupported ? "true" : "false",
            ["vector_bits"] = System.Runtime.Intrinsics.Vector512.IsHardwareAccelerated ? "512" : System.Runtime.Intrinsics.Vector256.IsHardwareAccelerated ? "256" : "128",
        };
        long total = MemInfo("MemTotal") ?? GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
        return new DeviceInfo
        {
            Provider = CpuExecutionProvider.ProviderName,
            Index = 0,
            Name = name,
            Kind = DeviceKind.Cpu,
            TotalMemory = total,
            Vendor = Vendor(),
            Driver = RuntimeInformation.FrameworkDescription,
            Capabilities = caps,
        };
    }

    private static string CpuName()
    {
        try
        {
            if (File.Exists("/proc/cpuinfo"))
            {
                foreach (var line in File.ReadLines("/proc/cpuinfo"))
                {
                    if (line.StartsWith("model name", StringComparison.Ordinal))
                    {
                        int i = line.IndexOf(':');
                        if (i >= 0)
                        {
                            return line[(i + 1)..].Trim();
                        }
                    }
                }
            }
        }
        catch { }
        return $"{RuntimeInformation.ProcessArchitecture} CPU";
    }

    private static string Vendor()
    {
        try
        {
            if (File.Exists("/proc/cpuinfo"))
            {
                foreach (var line in File.ReadLines("/proc/cpuinfo"))
                {
                    if (line.StartsWith("vendor_id", StringComparison.Ordinal))
                    {
                        int i = line.IndexOf(':');
                        if (i >= 0)
                        {
                            return line[(i + 1)..].Trim();
                        }
                    }
                }
            }
        }
        catch { }
        return "";
    }

    /// <summary>Reads a /proc/meminfo field in bytes (Linux); null elsewhere.</summary>
    private static long? MemInfo(string key)
    {
        try
        {
            if (!File.Exists("/proc/meminfo"))
            {
                return null;
            }

            foreach (var line in File.ReadLines("/proc/meminfo"))
            {
                if (line.StartsWith(key + ":", StringComparison.Ordinal))
                {
                    var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                    return long.Parse(parts[1]) * 1024;
                }
            }
        }
        catch { }
        return null;
    }

    public ICompiledSegment Compile(GraphSegment segment, CompileOptions options) => new CpuCompiledSegment(segment, options, this);

    // ----------------------------------------------------------------------------------------
    private static readonly HashSet<string> s_unary = new(StandardOps.UnaryNames);
    private static readonly HashSet<string> s_binary = new(StandardOps.BinaryNames);
    private static readonly HashSet<string> s_rows = ["RmsNorm", "L2Norm", "LayerNorm", "Softmax", "ReduceSum", "ReduceMax", "ArgMax"];

    private static bool IsStatic(Value v) => v.Kind is ValueKind.Weight or ValueKind.Constant;
    private static bool F32Like(Value v) => v.Type == DataType.F32 || (IsStatic(v) && Quant.FormatOf(v.Type) is not null);
    private static bool Plain(Value v) => !v.Type.IsQuantized && v.Type.BlockSize == 1 && v.Type.BlockBytes is 1 or 2 or 4 or 8;
    private static bool IsIdx(Value v) => v.Type == DataType.I32 || v.Type == DataType.I64;
    private static bool IsKv(DataType t) => t == DataType.F32 || t == DataType.F16;

    private static OpSupport CheckSpatialAttrs(string op, Sharplight.Graph.Attributes a, bool strides, bool dilations)
    {
        if (a.Has("strides") && (a.Get<long[]>("strides") is not { Length: 2 } st || st.Any(v => v < 1)))
        {
            return OpSupport.No($"{op}: strides must be 2 positive values");
        }

        if (a.Has("dilations") && (a.Get<long[]>("dilations") is not { Length: 2 } dl || dl.Any(v => v < 1)))
        {
            return OpSupport.No($"{op}: dilations must be 2 positive values");
        }

        if (a.Has("pads") && (a.Get<long[]>("pads") is not { Length: 4 } pd || pd.Any(v => v < 0)))
        {
            return OpSupport.No($"{op}: pads must be 4 non-negative values (top, left, bottom, right)");
        }

        return OpSupport.Yes;
    }

    public OpSupport Supports(Node node)
    {
        string op = node.Op.Name;
        var ins = node.Inputs;
        OpSupport AllF32(string what = "inputs")
        {
            foreach (var v in ins)
            {
                if (!F32Like(v))
                {
                    return OpSupport.No($"{op}: {what} must be f32 (or a convertible weight), got {v.Type} for {v.Name}");
                }
            }

            return OpSupport.Yes;
        }
        if (s_unary.Contains(op) || op == "Scale" || s_binary.Contains(op) || s_rows.Contains(op))
        {
            return AllF32();
        }

        switch (op)
        {
            case "Clip":
                return AllF32();
            case "Conv":
            {
                if (ins[0].Type != DataType.F32)
                    {
                        return OpSupport.No($"Conv input must be f32, got {ins[0].Type}");
                    }

                    var w = ins[1];
                if (!IsStatic(w))
                    {
                        return OpSupport.No("Conv weight must be a weight or constant (not an activation)");
                    }

                    var fmt = Quant.FormatOf(w.Type);
                if (fmt is null)
                    {
                        return OpSupport.No($"Conv weight type {w.Type} is not supported by the CPU provider");
                    }

                    long welems = w.Shape.Dims.Aggregate(1L, (acc, d) => acc * d.Fixed);
                if (welems % Quant.BlockSize(fmt.Value) != 0)
                    {
                        return OpSupport.No($"Conv weight has {welems} elements, not a multiple of the {w.Type} block size");
                    }

                    if (ins.Length > 2 && !F32Like(ins[2]))
                    {
                        return OpSupport.No($"Conv bias must be f32 (or a convertible weight), got {ins[2].Type}");
                    }

                    var a = node.Attributes;
                long g = a.Get("groups", 1L);
                if (g < 1 || ins[0].Shape.Dims[1].Fixed % g != 0 || w.Shape.Dims[0].Fixed % g != 0)
                    {
                        return OpSupport.No($"Conv groups={g} does not divide the channel counts");
                    }

                    if (ins[0].Shape.Dims[1].IsSymbolic || ins[0].Shape.Dims[2].IsSymbolic || ins[0].Shape.Dims[3].IsSymbolic)
                    {
                        return OpSupport.No("Conv with symbolic C/H/W");
                    }

                    return CheckSpatialAttrs(op, a, strides: true, dilations: true);
            }
            case "Pool":
            {
                if (ins[0].Type != DataType.F32)
                    {
                        return OpSupport.No($"Pool input must be f32, got {ins[0].Type}");
                    }

                    var a = node.Attributes;
                string mode = a.Get("mode", "max");
                if (mode != "max" && mode != "avg")
                    {
                        return OpSupport.No($"Pool mode '{mode}' is not supported (max | avg)");
                    }

                    if (!a.Has("kernel") || a.Get<long[]>("kernel").Length != 2)
                    {
                        return OpSupport.No("Pool needs a 2-element kernel attribute");
                    }

                    if (ins[0].Shape.Dims[2].IsSymbolic || ins[0].Shape.Dims[3].IsSymbolic)
                    {
                        return OpSupport.No("Pool with symbolic H/W");
                    }

                    if (mode == "avg" && a.Get("dilations", new long[] { 1, 1 }).Any(d => d != 1))
                    {
                        return OpSupport.No("avg Pool with dilations != 1");
                    }

                    return CheckSpatialAttrs(op, a, strides: true, dilations: true);
            }
            case "Reduce":
            {
                if (ins[0].Type != DataType.F32)
                    {
                        return OpSupport.No($"Reduce input must be f32, got {ins[0].Type}");
                    }

                    string mode = node.Attributes.Get("mode", "sum");
                if (mode is not ("sum" or "mean" or "max" or "min" or "prod" or "l1" or "l2" or "sumsquare" or "logsumexp"))
                    {
                        return OpSupport.No($"Reduce mode '{mode}' is not supported (sum | mean | max | min | prod | l1 | l2 | sumsquare | logsumexp)");
                    }

                    int r = ins[0].Rank;
                foreach (var ax in node.Attributes.Get<long[]>("axes"))
                    {
                        if (ax < -r || ax >= r)
                        {
                            return OpSupport.No($"Reduce axis {ax} out of range for rank {r}");
                        }
                    }

                    return OpSupport.Yes;
            }
            case "BatchMatMul":
                return AllF32();
            case "Pad":
            {
                if (ins[0].Type != DataType.F32)
                    {
                        return OpSupport.No($"Pad input must be f32, got {ins[0].Type}");
                    }

                    string mode = node.Attributes.Get("mode", "constant");
                if (mode is not ("constant" or "reflect" or "edge"))
                    {
                        return OpSupport.No($"Pad mode '{mode}' is not supported (constant | reflect | edge)");
                    }

                    return OpSupport.Yes;
            }
            case "Resize":
            {
                if (ins[0].Type != DataType.F32)
                    {
                        return OpSupport.No($"Resize input must be f32, got {ins[0].Type}");
                    }

                    var a = node.Attributes;
                string mode = a.Get("mode", "nearest"), coord = a.Get("coord", "half_pixel"), nm = a.Get("nearest_mode", "round_prefer_floor");
                if (mode is not ("nearest" or "linear"))
                    {
                        return OpSupport.No($"Resize mode '{mode}' is not supported (nearest | linear)");
                    }

                    if (coord is not ("half_pixel" or "align_corners" or "asymmetric" or "pytorch_half_pixel"))
                    {
                        return OpSupport.No($"Resize coord mode '{coord}' is not supported");
                    }

                    if (mode == "nearest" && nm is not ("round_prefer_floor" or "round_prefer_ceil" or "floor" or "ceil"))
                    {
                        return OpSupport.No($"Resize nearest_mode '{nm}' is not supported");
                    }

                    if (ins[0].Shape.Dims[2].IsSymbolic || ins[0].Shape.Dims[3].IsSymbolic)
                    {
                        return OpSupport.No("Resize with symbolic H/W");
                    }

                    return OpSupport.Yes;
            }
            case "Where":
            {
                if (!(ins[0].Type == DataType.F32 || ins[0].Type == DataType.Bool || ins[0].Type == DataType.I32 || ins[0].Type == DataType.U8))
                    {
                        return OpSupport.No($"Where condition must be an f32/bool/i32 mask, got {ins[0].Type}");
                    }

                    if (ins[1].Type != ins[2].Type)
                    {
                        return OpSupport.No("Where branches must share a type");
                    }

                    return Plain(ins[1]) ? OpSupport.Yes : OpSupport.No($"Where on {ins[1].Type}");
            }
            case "Expand" or "Tile":
                return Plain(ins[0]) ? OpSupport.Yes : OpSupport.No($"{op} on {ins[0].Type}");
            case "GatherAxis" or "GatherElements" or "GatherND":
                if (!Plain(ins[0]))
                {
                    return OpSupport.No($"{op} data type {ins[0].Type} is not supported");
                }

                if (!IsIdx(ins[1]))
                {
                    return OpSupport.No($"{op} indices must be i32 or i64, got {ins[1].Type}");
                }

                return OpSupport.Yes;
            case "ScatterElements" or "ScatterND":
            {
                if (!(ins[0].Type == DataType.F32 || ins[0].Type == DataType.I32 || ins[0].Type == DataType.I64))
                    {
                        return OpSupport.No($"{op} data must be f32, i32 or i64, got {ins[0].Type}");
                    }

                    if (!IsIdx(ins[1]))
                    {
                        return OpSupport.No($"{op} indices must be i32 or i64, got {ins[1].Type}");
                    }

                    if (ins[2].Type != ins[0].Type)
                    {
                        return OpSupport.No($"{op} updates must have the data type");
                    }

                    string red = node.Attributes.Get("reduction", "none");
                return red is "none" or "add" or "mul" or "max" or "min" ? OpSupport.Yes : OpSupport.No($"{op} reduction '{red}' is not supported");
            }
            case "CumSum" or "ArgReduce" or "TopK" or "GridSample":
                return AllF32();
            case "GroupNorm":
            {
                var s = AllF32(); if (!s.IsSupported)
                    {
                        return s;
                    }

                    long g = node.Attributes.Get<long>("groups");
                return g >= 1 && !ins[0].Shape.Dims[1].IsSymbolic && ins[0].Shape.Dims[1].Fixed % g == 0 ? OpSupport.Yes : OpSupport.No($"GroupNorm groups={g} does not divide the channel count");
            }
            case "Trilu":
                return Plain(ins[0]) && ins[0].Rank >= 2 ? OpSupport.Yes : OpSupport.No("Trilu needs rank >= 2 of a plain type");
            case "OneHot":
                return IsIdx(ins[0]) ? OpSupport.Yes : OpSupport.No($"OneHot indices must be i32 or i64, got {ins[0].Type}");
            case "ConvTranspose":
            {
                if (ins[0].Type != DataType.F32)
                    {
                        return OpSupport.No($"ConvTranspose input must be f32, got {ins[0].Type}");
                    }

                    var w = ins[1];
                if (!IsStatic(w))
                    {
                        return OpSupport.No("ConvTranspose weight must be a weight or constant");
                    }

                    var fmt = Quant.FormatOf(w.Type);
                if (fmt is null)
                    {
                        return OpSupport.No($"ConvTranspose weight type {w.Type} is not supported by the CPU provider");
                    }

                    long welems = w.Shape.Dims.Aggregate(1L, (acc, d) => acc * d.Fixed);
                if (welems % Quant.BlockSize(fmt.Value) != 0)
                    {
                        return OpSupport.No($"ConvTranspose weight has {welems} elements, not a multiple of the {w.Type} block size");
                    }

                    if (ins.Length > 2 && !F32Like(ins[2]))
                    {
                        return OpSupport.No($"ConvTranspose bias must be f32, got {ins[2].Type}");
                    }

                    long g = node.Attributes.Get("groups", 1L);
                if (g < 1 || ins[0].Shape.Dims[1].Fixed % g != 0)
                    {
                        return OpSupport.No($"ConvTranspose groups={g} does not divide the input channels");
                    }

                    return CheckSpatialAttrs(op, node.Attributes, strides: true, dilations: true);
            }
            case "Reshape" or "Slice" or "Permute": return OpSupport.Yes;
            case "Contiguous" or "Transfer":
                return ins[0].Type.IsQuantized ? OpSupport.No($"{op} of quantized data") : OpSupport.Yes;
            case "Concat":
                if (ins.Any(v => v.Type.IsQuantized))
                {
                    return OpSupport.No("Concat of quantized data");
                }

                if (ins.Any(v => v.Type != ins[0].Type))
                {
                    return OpSupport.No("Concat inputs must share a type");
                }
                {
                    int axis = node.Attributes.Get<int>("axis"); if (axis < 0)
                    {
                        axis += ins[0].Rank;
                    }

                    if (axis == 0 && ins[0].Shape.Dims[0].IsSymbolic)
                    {
                        return OpSupport.No("Concat along the symbolic axis");
                    }
                }
                return OpSupport.Yes;
            case "Cast":
            {
                var src = ins[0].Type; var dst = node.Output.Type;
                static bool Ok(DataType t) => t == DataType.F32 || t == DataType.F16 || t == DataType.BF16 || t == DataType.I32 || t == DataType.I64 || t == DataType.I8 || t == DataType.U8 || t == DataType.Bool;
                return Ok(src) && Ok(dst) ? OpSupport.Yes : OpSupport.No($"Cast {src} -> {dst}");
            }
            case "MatMul":
            {
                if (!F32Like(ins[0]))
                    {
                        return OpSupport.No($"MatMul activations must be f32, got {ins[0].Type}");
                    }

                    var w = ins[1];
                if (Quant.FormatOf(w.Type) is null)
                    {
                        return OpSupport.No($"MatMul weight type {w.Type} is not supported by the CPU provider");
                    }

                    if (!IsStatic(w) && w.Type != DataType.F32)
                    {
                        return OpSupport.No("MatMul with a non-f32 activation as the weight");
                    }

                    long k = w.Shape.Dims[1].Fixed;
                if (k % Quant.BlockSize(Quant.FormatOf(w.Type)!.Value) != 0)
                    {
                        return OpSupport.No($"K={k} is not a multiple of the {w.Type} block size");
                    }

                    return OpSupport.Yes;
            }
            case "MatMulId":
            {
                if (!F32Like(ins[0]))
                    {
                        return OpSupport.No($"MatMulId activations must be f32, got {ins[0].Type}");
                    }

                    var w = ins[1];
                if (Quant.FormatOf(w.Type) is null)
                    {
                        return OpSupport.No($"MatMulId weight type {w.Type} is not supported by the CPU provider");
                    }

                    if (!IsStatic(w) && w.Type != DataType.F32)
                    {
                        return OpSupport.No("MatMulId with a non-f32 activation as the weight");
                    }

                    if (ins[2].Type != DataType.I32)
                    {
                        return OpSupport.No("MatMulId expert indices must be i32");
                    }

                    long k = w.Shape.Dims[2].Fixed;
                if (k % Quant.BlockSize(Quant.FormatOf(w.Type)!.Value) != 0)
                    {
                        return OpSupport.No($"K={k} is not a multiple of the {w.Type} block size");
                    }

                    return OpSupport.Yes;
            }
            case "Gather":
            {
                var t = ins[0];
                if (Quant.FormatOf(t.Type) is null)
                    {
                        return OpSupport.No($"Gather table type {t.Type} is not supported by the CPU provider");
                    }

                    if (!IsStatic(t) && t.Type.IsQuantized)
                    {
                        return OpSupport.No("Gather from a quantized activation");
                    }

                    if (ins[1].Type != DataType.I32)
                    {
                        return OpSupport.No("Gather indices must be i32");
                    }

                    return OpSupport.Yes;
            }
            case "Rope":
                if (ins[0].Type != DataType.F32 && !IsStatic(ins[0]))
                {
                    return OpSupport.No("Rope input must be f32");
                }

                if (ins[0].Rank != 3)
                {
                    return OpSupport.No("Rope input must be [T, H, D]");
                }

                if (ins[1].Type != DataType.I32)
                {
                    return OpSupport.No("Rope positions must be i32");
                }

                return OpSupport.Yes;
            case "CacheWrite":
                if (!(ins[0].Type == DataType.F32 || ins[0].Type == DataType.F16 || ins[0].Type == DataType.BF16))
                {
                    return OpSupport.No($"KV cache type {ins[0].Type} is not supported by the CPU provider");
                }

                return ins[1].Type == DataType.F32 ? OpSupport.Yes : OpSupport.No("CacheWrite source must be f32");
            case "CachedAttention":
                if (ins[0].Type != DataType.F32)
                {
                    return OpSupport.No("CachedAttention q must be f32");
                }

                if (node.Attributes.Get("sinks", false) && ins[3].Type != DataType.F32)
                {
                    return OpSupport.No("CachedAttention sinks must be f32");
                }

                if (!IsKv(ins[1].Type) || !IsKv(ins[2].Type))
                {
                    return OpSupport.No($"KV cache type {ins[1].Type}/{ins[2].Type} is not supported by the CPU provider (f32/f16 only)");
                }

                return OpSupport.Yes;
            case "Attention":
                if (ins[0].Type != DataType.F32)
                {
                    return OpSupport.No("Attention q must be f32");
                }

                if (!IsKv(ins[1].Type) || !IsKv(ins[2].Type))
                {
                    return OpSupport.No("Attention k/v must be f32 or f16");
                }

                if (ins.Length > 3 && !(ins[3].Type == DataType.F32 && ins[3].Rank == 2) && !(ins[3].Type == DataType.I32 && ins[3].Rank == 2 && ins[3].Shape[1] == 2))
                {
                    return OpSupport.No("Attention fourth input must be an f32 [T, S] mask or i32 [T, 2] spans");
                }

                return OpSupport.Yes;
            case "SequencePool":
                if (ins[0].Type != DataType.F32)
                {
                    return OpSupport.No("SequencePool input must be f32");
                }

                if (ins[1].Type != DataType.I32 || ins[1].Rank != 2 || ins[1].Shape[1] != 2)
                {
                    return OpSupport.No("SequencePool spans must be i32 [B, 2]");
                }

                if (node.Attributes.Get("mode", "mean") is not ("mean" or "sum" or "max" or "first" or "last"))
                {
                    return OpSupport.No("SequencePool mode must be mean | sum | max | first | last");
                }

                return OpSupport.Yes;
            case "CausalConv1d":
                if (ins[0].Type != DataType.F32 || ins[2].Type != DataType.F32)
                {
                    return OpSupport.No("CausalConv1d needs f32 activations and state");
                }

                bool convBias = node.Attributes.Get("bias", false) || (ins.Length > 3 && ins[3].Type != DataType.I32);
                return F32Like(ins[1]) && (!convBias || F32Like(ins[3])) ? OpSupport.Yes : OpSupport.No("CausalConv1d weight/bias must be f32-convertible");
            case "SelectiveScan":
                if (ins[0].Type != DataType.F32 || ins[1].Type != DataType.F32 || ins[4].Type != DataType.F32 || ins[5].Type != DataType.F32 || ins[7].Type != DataType.F32)
                {
                    return OpSupport.No("SelectiveScan needs f32 x/dt/B/C and state");
                }

                return F32Like(ins[2]) && F32Like(ins[3]) && F32Like(ins[6]) ? OpSupport.Yes : OpSupport.No("SelectiveScan dt_bias/A/D must be f32-convertible");
            case "GatedDeltaRule":
                foreach (var v in ins.Take(6))
                {
                    if (v.Type != DataType.F32)
                    {
                        return OpSupport.No($"GatedDeltaRule needs f32 inputs and state, got {v.Type}");
                    }
                }

                if (ins.Length > 6 && ins[6].Type != DataType.I32)
                {
                    return OpSupport.No("GatedDeltaRule sequence runs must be i32");
                }

                return OpSupport.Yes;
        }
        return OpSupport.No($"op '{op}' is not implemented by the CPU provider");
    }

    public void Dispose() { }
}
