using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;
using Sharplight.Cuda.Interop;
using Sharplight.Runtime;
using Sharplight.Shaders;

namespace Sharplight.Cuda;

/// <summary>
/// Compiles the CUDA C++ Slang generates into PTX for one virtual architecture (<c>compute_XY</c>) with NVRTC, the portable code of the
/// CUDA provider. The block size header of the source (<see cref="SlangCompiler.CudaThreadsHeader"/>) is kept as the first line of the PTX.
/// Kernels whose shared arrays exceed CUDA's 48 KB static limit are moved to dynamic shared memory (<see cref="SharedMemoryRewriter"/>);
/// their size follows the block size as a second header line (<see cref="CudaPtx.SharedHeader"/>), and the tensor maps of their
/// parameters a third (<see cref="SlangCompiler.CudaTensorMapsHeader"/>).
/// The floating-point headers the source may include (<c>cuda_fp16.h</c>, <c>cuda_bf16.h</c>, <c>cuda_fp8.h</c>) are passed from the files
/// shipped next to NVRTC, so no CUDA toolkit is needed. Thread-safe.
/// </summary>
internal sealed partial class NvrtcTranslator : KernelTranslator
{
    /// <summary>Version of the options and the PTX post-processing; part of every cache key.</summary>
    private const string OptionsVersion = "7";

    /// <summary>The headers NVRTC needs for the half, bfloat16 and 8-bit float types (redistributable, see docs/cuda.md).</summary>
    private static readonly string[] s_headerNames = ["cuda_fp16.h", "cuda_fp16.hpp", "cuda_bf16.h", "cuda_bf16.hpp", "cuda_fp8.h", "cuda_fp8.hpp"];

    private static readonly Lazy<IReadOnlyList<(string Name, string Text)>> s_headers = new(LoadHeaders);
    private static long s_ticks;
    private static int s_compiled;

    /// <param name="computeCapability">Compute capability the PTX is generated for (e.g. 120 for 12.0).</param>
    public NvrtcTranslator(int computeCapability)
    {
        ComputeCapability = computeCapability;
    }

    /// <summary>The compute capability (major * 10 + minor) the PTX targets.</summary>
    public int ComputeCapability { get; }

    /// <inheritdoc />
    public override ShaderTarget Target => ShaderTarget.Ptx;

    /// <inheritdoc />
    public override ShaderTarget SourceTarget => ShaderTarget.CudaSource;

    /// <inheritdoc />
    public override string Id => $"nvrtc|compute_{ComputeCapability}|{OptionsVersion}{(DiagnosticSwitches.DependentTrigger ? "" : "|pdl-wait")}{(DiagnosticSwitches.MinBlocks is { } bounds ? "|bounds=" + bounds : "")}{(DiagnosticSwitches.NvrtcSass is { } sass ? "|sass=" + sass : "")}";

    /// <summary>Kernels this process compiled with NVRTC and the time spent (diagnostics).</summary>
    public static (int Kernels, TimeSpan Time) Statistics => (Volatile.Read(ref s_compiled), Stopwatch.GetElapsedTime(0, Volatile.Read(ref s_ticks)));

    /// <summary>The NVRTC version, or null when NVRTC is not installed.</summary>
    public static unsafe Version? NvrtcVersion
    {
        get
        {
            CudaLibraries.EnsureResolver();
            try
            {
                int major;
                int minor;
                return NvrtcApi.nvrtcVersion(&major, &minor) == 0 ? new Version(major, minor) : null;
            }
            catch (DllNotFoundException)
            {
                return null;
            }
        }
    }

    /// <inheritdoc />
    /// <exception cref="KernelCompilationException">NVRTC rejected the source, or NVRTC is not installed.</exception>
    public override byte[] Translate(string name, string entryPoint, byte[] code)
    {
        long start = Stopwatch.GetTimestamp();
        string source = Encoding.UTF8.GetString(code);
        int headerEnd = source.IndexOf('\n');
        string threadsLine = source.StartsWith(SlangCompiler.CudaThreadsHeader, StringComparison.Ordinal) && headerEnd > 0 ? source[..(headerEnd + 1)] : "";
        // Kernels written in CUDA C++ (SlangCompiler.CompileCuda) state their dynamic shared memory in a second header line.
        int declaredShared = 0;
        string mapsLine = "";
        if (headerEnd > 0 && source.AsSpan(headerEnd + 1).StartsWith(SlangCompiler.CudaSharedHeader, StringComparison.Ordinal))
        {
            int valueStart = headerEnd + 1 + SlangCompiler.CudaSharedHeader.Length;
            int sharedEnd = source.IndexOf('\n', valueStart);
            declaredShared = int.Parse(source.AsSpan(valueStart, sharedEnd - valueStart), CultureInfo.InvariantCulture);
            // The tensor maps of the parameters, if any, follow as a third header line.
            if (source.AsSpan(sharedEnd + 1).StartsWith(SlangCompiler.CudaTensorMapsHeader, StringComparison.Ordinal))
            {
                mapsLine = source[(sharedEnd + 1)..(source.IndexOf('\n', sharedEnd + 1) + 1)];
            }
        }

        byte[] ptx;
        byte[]? cubin = null;
        int dynamicShared = 0;
        bool dependent = false;
        try
        {
            source = WithMinBlocks(source, name, entryPoint, threadsLine);
            source = SharedMemoryRewriter.Rewrite(source, out dynamicShared);
            dynamicShared = Math.Max(dynamicShared, declaredShared);
            dependent = ComputeCapability >= DependentLaunchCapability && WaitForPrevious(ref source);
            string fileName = $"{Path.GetFileNameWithoutExtension(name)}_{entryPoint}.cu";
            ptx = Compile(source, fileName, Options(ComputeCapability, FamilySpecific(source)), cubin: false);
            if (CompilesSass(source, ComputeCapability))
            {
                cubin = Compile(source, fileName, Options(ComputeCapability, FamilySpecific(source), real: true), cubin: true);
            }
        }
        catch (DllNotFoundException e)
        {
            throw new KernelCompilationException($"Kernel {name}:{entryPoint} needs NVRTC, which is not installed (add the Sharplight.Cuda.Runtime package, or a precompiled model): {e.Message}");
        }

        Interlocked.Add(ref s_ticks, Stopwatch.GetTimestamp() - start);
        Interlocked.Increment(ref s_compiled);
        if (DiagnosticSwitches.DumpSourceDirectory is { } dump)
        {
            Directory.CreateDirectory(dump);
            string stem = Path.Combine(dump, $"{Path.GetFileNameWithoutExtension(name)}_{entryPoint}_{Environment.CurrentManagedThreadId}_{start}");
            File.WriteAllText(stem + ".cu", source);
            File.WriteAllBytes(stem + ".ptx", ptx);
        }

        string sharedLine = dynamicShared > 0 ? $"{CudaPtx.SharedHeader}{dynamicShared}\n" : "";
        string dependentLine = dependent ? CudaPtx.DependentHeader + "\n" : "";
        string cubinLine = cubin is null ? "" : $"{CudaPtx.CubinHeader}sm_{ComputeCapability} {Convert.ToBase64String(cubin)}\n";
        return [.. Encoding.UTF8.GetBytes(threadsLine + sharedLine + mapsLine + dependentLine + cubinLine), .. ptx];
    }

    /// <summary>
    /// Launch bounds of the generated kernels by default (see <see cref="WithMinBlocks"/>): the expert-grouped skinny GEMMs (gemm_sk of
    /// the MoE sources) as one 128-thread block per SM. Without the bounds ptxas schedules them for full occupancy, and their weight
    /// stream stalls: gpt-oss-20b decode steps at 16 / 64 sequences 12.04 / 17.69 → 10.05 / 16.22 ms on the RTX PRO 4500 (the same
    /// kernels' SPIR-V forms take 10.0 / 16.7 ms on the NVIDIA Vulkan driver; benchmarks.md section 61). The dense skinny GEMMs keep
    /// none (Qwen3-8B-FP8 at 16 sequences +0.2 ms with them). The defaults reach the cache keys only through <see cref="OptionsVersion"/>
    /// (unlike <c>SHARPLIGHT_CUDA_MIN_BLOCKS</c>, which is in <see cref="Id"/>): changing them changes the PTX of kernels whose source
    /// does not change, so it needs a version bump, or cached and precompiled PTX of the old bounds would still be served.
    /// </summary>
    private const string DefaultMinBlocks = "matmul_moe_dual:gemm_sk:1;matmul_moe_epi:gemm_sk:1;matmul_moe:gemm_sk:1";

    /// <summary>
    /// <c>SHARPLIGHT_CUDA_MIN_BLOCKS=file:entry:blocks[;...]</c> (replaces the defaults; `none` = no bounds): kernels whose file name contains <c>file</c> and whose entry
    /// is <c>entry</c> get <c>__launch_bounds__(threads, blocks)</c>, so ptxas caps their registers for that many resident blocks per SM.
    /// </summary>
    private static string WithMinBlocks(string source, string name, string entryPoint, string threadsLine)
    {
        string rules = DiagnosticSwitches.MinBlocks ?? DefaultMinBlocks;
        if (threadsLine.Length == 0)
        {
            return source;
        }

        foreach (var rule in rules.Split(';', StringSplitOptions.RemoveEmptyEntries))
        {
            var parts = rule.Split(':');
            if (parts.Length == 3 && name.Contains(parts[0], StringComparison.Ordinal) && entryPoint == parts[1])
            {
                int threads = int.Parse(threadsLine[SlangCompiler.CudaThreadsHeader.Length..].Split(' ')[0], CultureInfo.InvariantCulture);
                return KernelEntryPattern().Replace(source, m => m.Value.Replace("__global__", $"__global__ __launch_bounds__({threads}, {parts[2]})", StringComparison.Ordinal), 1);
            }
        }

        return source;
    }

    /// <summary>Compute capability from which kernels wait for the previous launch themselves (<c>griddepcontrol</c>, PTX 7.8, sm_90).</summary>
    internal const int DependentLaunchCapability = 90;

    /// <summary>
    /// Starts the body of every kernel (<c>extern "C" __global__</c>) of <paramref name="source"/> with <c>griddepcontrol.wait</c> (the
    /// kernel waits for the completion and the memory of the launch it depends on) and <c>griddepcontrol.launch_dependents</c> (the next
    /// launch may start its blocks, which then wait the same way). Without a programmatic dependency both are no-ops, so the PTX serves
    /// plain launches alike; with one (<see cref="DeviceFeatures.DependentLaunches"/>) the next kernel's launch and block setup overlap
    /// this kernel's last blocks. Returns false when the source has no kernel entry it recognizes.
    /// </summary>
    /// <remarks>
    /// A kernel whose source issues <c>griddepcontrol.wait</c> itself (after reads that no earlier launch writes, such as weights)
    /// gets only the trigger at its entry.
    /// </remarks>
    internal static bool WaitForPrevious(ref string source)
    {
        bool waitsItself = source.Contains("griddepcontrol.wait", StringComparison.Ordinal);
        string prologue = waitsItself
            ? DiagnosticSwitches.DependentTrigger ? "\n    asm volatile(\"griddepcontrol.launch_dependents;\" ::: \"memory\");\n" : "\n"
            : DiagnosticSwitches.DependentTrigger
                ? "\n    asm volatile(\"griddepcontrol.wait;\\n\\tgriddepcontrol.launch_dependents;\" ::: \"memory\");\n"
                : "\n    asm volatile(\"griddepcontrol.wait;\" ::: \"memory\");\n";
        var text = new StringBuilder(source.Length + 256);
        int at = 0, count = 0;
        foreach (Match entry in KernelEntryPattern().Matches(source))
        {
            int body = entry.Index + entry.Length;
            text.Append(source, at, body - at).Append(prologue);
            at = body;
            count++;
        }

        if (count == 0)
        {
            return false;
        }

        source = text.Append(source, at, source.Length - at).ToString();
        return true;
    }

    [GeneratedRegex(@"extern\s+""C""\s+__global__[^{;]*\)\s*\{")]
    private static partial Regex KernelEntryPattern();

    /// <summary>
    /// The NVRTC options of a kernel: PTX for the virtual architecture, no warnings (the Slang prelude has many). The half, bfloat16 and
    /// 8-bit float headers are always on: Slang lowers <c>f16tof32</c> to the <c>cuda_fp16.h</c> intrinsics but enables the header only
    /// for kernels with <c>half</c> values, and its cooperative matrices need the type of every fragment.
    /// </summary>
    /// <remarks>
    /// <paramref name="familySpecific"/> targets the family-specific virtual architecture (<c>compute_XYf</c>, NVRTC 12.9 and later),
    /// whose PTX runs on the GPUs of the same major version only: the instructions of <see cref="FamilySpecific"/> kernels exist there only.
    /// </remarks>
    internal static string[] Options(int computeCapability, bool familySpecific = false, bool real = false) =>
        [$"-arch={(real ? "sm" : "compute")}_{computeCapability}{(familySpecific ? "f" : "")}", "-default-device", "-std=c++17", "-w", "-DSLANG_CUDA_ENABLE_HALF", "-DSLANG_CUDA_ENABLE_BF16", "-DSLANG_CUDA_ENABLE_FP8"];

    /// <summary>
    /// True when <paramref name="source"/> is specialized with <c>SL_FAMILY_ARCH</c> (the block-scaled MMA of <c>gemm_fp8_tma.cu</c>,
    /// <see cref="DeviceFeatures.BlockScaledFp8"/>): it uses instructions of the family-specific architecture, which the portable
    /// <c>compute_XY</c> PTX does not have.
    /// </summary>
    internal static bool FamilySpecific(string source) => source.Contains("\n#define SL_FAMILY_ARCH 1\n", StringComparison.Ordinal);

    /// <summary>
    /// True when NVRTC also compiles <paramref name="source"/> to machine code for the real architecture (<c>sm_XY</c>), which the device
    /// then loads instead of letting the driver compile the PTX: kernels that define <c>SL_NVRTC_SASS</c> (<see cref="SassMarker"/>), on
    /// GPUs of compute capability 12.x. The driver's PTX compiler is whatever the installed driver brings; NVRTC's is the shipped release
    /// (Sharplight.Cuda.Runtime), whose instruction schedule is the one measured. The FP8 TMA GEMM (<c>gemm_fp8_tma.cu</c>) asks for it:
    /// on the RTX PRO 4500 (driver 595, CUDA 13.2) Qwen3-8B-FP8's prompt steps take 2-5 % less time with NVRTC 13.0's code of its 9-warp
    /// form (benchmarks.md section 92), while most other kernels are as fast or faster from the driver. (The driver also drops
    /// <c>setmaxnreg</c> from family-specific PTX: its cubins target <c>sm_120</c>.) The PTX stays in the code for other GPUs and as the fallback.
    /// <c>SHARPLIGHT_CUDA_NVRTC_SASS=0</c> turns it off, <c>all</c> applies it to every kernel, <c>if:text</c> to the kernels whose source
    /// contains the text (<c>\n</c> for a line break, <c>|</c> between alternatives; diagnostics).
    /// </summary>
    internal static bool CompilesSass(string source, int computeCapability) => DiagnosticSwitches.NvrtcSass switch
    {
        "0" => false,
        "all" => true,
        ['i', 'f', ':', ..] text => text[3..].Split('|').Any(t => source.Contains(t.Replace("\\n", "\n", StringComparison.Ordinal), StringComparison.Ordinal)),
        _ => computeCapability / 10 == 12 && source.Contains(SassMarker, StringComparison.Ordinal),
    };

    /// <summary>The line of a kernel source that asks for machine code from NVRTC (<see cref="CompilesSass"/>).</summary>
    internal const string SassMarker = "\n#define SL_NVRTC_SASS 1\n";

    /// <summary>Compiles CUDA C++ with NVRTC into PTX (or SASS for one real architecture when <paramref name="cubin"/> is set).</summary>
    /// <exception cref="KernelCompilationException">NVRTC rejected the source.</exception>
    /// <exception cref="DllNotFoundException">NVRTC is not installed.</exception>
    internal static unsafe byte[] Compile(string source, string fileName, IReadOnlyList<string> options, bool cubin)
    {
        CudaLibraries.EnsureResolver();
        var headers = s_headers.Value;
        var pinned = new List<GCHandle>();
        try
        {
            byte* Utf8(string text)
            {
                var handle = GCHandle.Alloc(Encoding.UTF8.GetBytes(text + "\0"), GCHandleType.Pinned);
                pinned.Add(handle);
                return (byte*)handle.AddrOfPinnedObject();
            }

            var headerTexts = stackalloc byte*[headers.Count];
            var headerNames = stackalloc byte*[headers.Count];
            for (int i = 0; i < headers.Count; i++)
            {
                headerTexts[i] = Utf8(headers[i].Text);
                headerNames[i] = Utf8(headers[i].Name);
            }

            nint program;
            NvrtcApi.Check(NvrtcApi.nvrtcCreateProgram(&program, Utf8(source), Utf8(fileName), headers.Count, headerTexts, headerNames), "create program");
            try
            {
                var optionPointers = stackalloc byte*[options.Count];
                for (int i = 0; i < options.Count; i++)
                {
                    optionPointers[i] = Utf8(options[i]);
                }

                int result = NvrtcApi.nvrtcCompileProgram(program, options.Count, optionPointers);
                if (result != 0)
                {
                    throw new KernelCompilationException($"NVRTC rejected {fileName} ({result}):\n{Log(program)}");
                }

                nuint size;
                byte[] output;
                if (cubin)
                {
                    NvrtcApi.Check(NvrtcApi.nvrtcGetCUBINSize(program, &size), "cubin size");
                    output = new byte[size];
                    fixed (byte* p = output)
                    {
                        NvrtcApi.Check(NvrtcApi.nvrtcGetCUBIN(program, p), "get cubin");
                    }

                    return output;
                }

                NvrtcApi.Check(NvrtcApi.nvrtcGetPTXSize(program, &size), "PTX size");
                output = new byte[size];
                fixed (byte* p = output)
                {
                    NvrtcApi.Check(NvrtcApi.nvrtcGetPTX(program, p), "get PTX");
                }

                // The size includes the terminating NUL; the module loader appends one again.
                return output.AsSpan().TrimEnd((byte)0).ToArray();
            }
            finally
            {
                NvrtcApi.nvrtcDestroyProgram(&program);
            }
        }
        finally
        {
            foreach (var handle in pinned)
            {
                handle.Free();
            }
        }
    }

    private static unsafe string Log(nint program)
    {
        nuint size;
        if (NvrtcApi.nvrtcGetProgramLogSize(program, &size) != 0 || size == 0)
        {
            return "";
        }

        var log = new byte[size];
        fixed (byte* p = log)
        {
            NvrtcApi.nvrtcGetProgramLog(program, p);
        }

        return Encoding.UTF8.GetString(log).TrimEnd('\0');
    }

    /// <summary>The floating-point headers next to NVRTC (missing ones are left out: only kernels that include them fail).</summary>
    private static IReadOnlyList<(string Name, string Text)> LoadHeaders()
    {
        // Loading NVRTC fixes the directory the headers ship in.
        _ = NvrtcVersion;
        var directories = CudaLibraries.NvrtcDirectory is { } nvrtc ? [nvrtc, .. CudaLibraries.RuntimeDirectories()] : CudaLibraries.RuntimeDirectories();
        var headers = new List<(string, string)>();
        foreach (string header in s_headerNames)
        {
            string? path = directories.Select(d => Path.Combine(d, header)).FirstOrDefault(File.Exists);
            if (path is not null)
            {
                headers.Add((header, File.ReadAllText(path)));
            }
        }

        return headers;
    }
}
