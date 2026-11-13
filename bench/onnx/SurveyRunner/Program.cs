// Throwaway survey harness: imports an ONNX model described by a reference manifest (same format as the ZooTests fixtures),
// runs it on cpu|vulkan, compares with the onnxruntime reference outputs and times it.
//   SurveyRunner <manifest.json> <cpu|vulkan> [--runs N] [--budget-s S] [--device D] [--batch B]
// Prints exactly one line starting with "RESULT " followed by a JSON object.
using System.Diagnostics;
using System.Text.Json;
using Sharplight.Cpu;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Onnx;
using Sharplight.Runtime;
using Sharplight.Tensors;
using Sharplight.Vulkan;

var argv = args.ToList();
string? Opt(string n, string? d = null) { int i = argv.IndexOf(n); return i >= 0 && i + 1 < argv.Count ? argv[i + 1] : d; }
string manifestPath = argv[0], providerName = argv[1];
int maxRuns = int.Parse(Opt("--runs", "30")!);
double budgetS = double.Parse(Opt("--budget-s", "4")!, System.Globalization.CultureInfo.InvariantCulture);
int devIdx = int.Parse(Opt("--device", "0")!);

var res = new Dictionary<string, object?> { ["manifest"] = manifestPath, ["provider"] = providerName, ["stage"] = "start" };
void Emit() { Console.Out.Flush(); Console.WriteLine("RESULT " + JsonSerializer.Serialize(res)); }

try
{
    using var doc = JsonDocument.Parse(File.ReadAllText(manifestPath));
    var r = doc.RootElement;
    string root = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models", "onnx");
    string name = Path.GetFileNameWithoutExtension(manifestPath);
    string modelPath = r.TryGetProperty("model", out var mp) ? Path.Combine(root, mp.GetString()!) : Path.Combine(root, name + ".onnx");
    string prefix = manifestPath[..^5];
    int batch = r.TryGetProperty("batches", out var bs) ? bs.EnumerateArray().First().GetInt32() : 1;
    if (Opt("--batch") is { } bo) batch = int.Parse(bo);
    var inputs = r.GetProperty("inputs").EnumerateArray().Select(e => (Name: e.GetProperty("name").GetString()!, Shape: e.GetProperty("shape").EnumerateArray().Select(x => x.GetInt64()).ToArray())).ToArray();
    var outNames = r.GetProperty("outputs").EnumerateArray().Select(e => e.GetProperty("name").GetString()!).ToArray();
    double rtol = r.TryGetProperty("rtol", out var rt) ? rt.GetDouble() : 1e-3, atol = r.TryGetProperty("atol", out var at) ? at.GetDouble() : 2e-4;
    string? mode = r.TryGetProperty("mode", out var md) && md.ValueKind == JsonValueKind.String ? md.GetString() : null;
    if (name == "yolov10n") mode = "scores6";   // ReferenceTests compares only the score column of this near-tied detection output
    double? rmsTol = r.TryGetProperty("rms", out var rm) && rm.ValueKind == JsonValueKind.Number ? rm.GetDouble() : null;
    bool fixBatch = r.TryGetProperty("fixbatch", out var fb) && fb.GetBoolean() || name.StartsWith("whisper");
    var consts = r.TryGetProperty("constants", out var cs) && cs.ValueKind == JsonValueKind.Object
        ? cs.EnumerateObject().ToDictionary(p => p.Name, p => p.Value.EnumerateArray().Select(x => x.GetDouble()).ToArray()) : new();
    res["model"] = modelPath; res["batch"] = batch; res["modelMB"] = new FileInfo(modelPath).Length >> 20;

    IExecutionProvider provider = providerName == "cpu" ? new CpuExecutionProvider() : new VulkanProvider();
    var av = provider.CheckAvailability();
    if (!av.IsAvailable) { res["stage"] = "provider"; res["error"] = av.Reason; Emit(); return 2; }
    using var device = provider.OpenDevice(provider.GetDevices()[devIdx]);
    res["device"] = provider.GetDevices()[devIdx].ToString();

    ComputeGraph? graph = null; ModelDefinition? def = null; CompiledModel? compiled = null;
    var sw = Stopwatch.StartNew();
    foreach (bool fix in fixBatch ? new[] { true } : new[] { false, true })
    {
        try
        {
            var o = new OnnxImportOptions();
            foreach (var (k, v) in consts) o.ConstantInputs[k] = v;
            foreach (var i in inputs) o.InputShapes[i.Name] = fix ? i.Shape.Select(d => d < 0 ? batch : d).ToArray() : i.Shape;
            res["stage"] = "import"; sw.Restart();
            var src = OnnxImporter.Open(modelPath, o);
            def = OnnxGraphAdapter.Build(src);
            graph = def.Graph(OnnxGraphAdapter.MainGraph);
            res["importMs"] = Math.Round(sw.Elapsed.TotalMilliseconds); res["nodes"] = graph.Nodes.Count; res["opset"] = src.Get("onnx.opset", 0L);
            res["stage"] = "compile"; sw.Restart();
            compiled = CompiledModel.Compile(graph, new PlacementOptions
            {
                Devices = [new DeviceSlot { Device = device }],
                Compile = new CompileOptions { SymbolBounds = new Dictionary<string, long> { [OnnxGraphAdapter.BatchSymbol] = batch } },
                ReservePerDevice = 256L << 20,
            });
            res["compileMs"] = Math.Round(sw.Elapsed.TotalMilliseconds); res["segments"] = compiled.Plan.Segments.Count;
            res["fixedBatch"] = fix;
            if (fix && !fixBatch) res["note"] = "symbolic batch failed (" + Trunc((string?)res.GetValueOrDefault("symbolicError")) + "); fixed batch used";
            break;
        }
        catch (Exception e) when (!fix && !fixBatch)
        {
            res["symbolicError"] = Trunc(e.Message); compiled?.Dispose(); compiled = null; graph = null;
        }
    }
    using var _c = compiled; 
    var run = new RunParameters(); run.Symbols[OnnxGraphAdapter.BatchSymbol] = batch;
    res["stage"] = "run";
    // inputs: manifest order includes pinned constants (not graph inputs); bins are indexed by the manifest position.
    for (int gi = 0; gi < graph!.Inputs.Count; gi++)
    {
        var gv = graph.Inputs[gi];
        int k = Array.FindIndex(inputs, x => x.Name == gv.Name);
        if (k < 0) { var rtIdx = Enumerable.Range(0, inputs.Length).Where(q => !consts.ContainsKey(inputs[q].Name)).ToArray(); k = rtIdx[gi]; }
        var f = ReadF32($"{prefix}.b{batch}.in{k}.bin");
        var bytes = FromF32(f, gv.Type);
        long want = gv.Type.ByteSize(gv.Shape.Elements(run.Symbols));
        if (bytes.Length != want) throw new InvalidOperationException($"input '{gv.Name}' {gv.Type} {gv.Shape}: have {bytes.Length} bytes, graph wants {want}");
        compiled!.SetInput(gv, bytes);
    }
    compiled!.Run(run);
    var outs = new List<(string Name, float[] Data)>();
    foreach (var (oname, ov) in graph.Outputs) outs.Add((oname, ReadOut(compiled, oname, ov, run)));
    // compare
    var cmps = new List<object>(); bool allOk = true; double worstRms = 0;
    for (int k = 0; k < outNames.Length; k++)
    {
        int gi = outs.FindIndex(o => o.Name == outNames[k]); if (gi < 0) gi = k;
        if (gi >= outs.Count) { cmps.Add(new { name = outNames[k], error = "output missing" }); allOk = false; continue; }
        var exp = ReadF32($"{prefix}.b{batch}.out{k}.bin"); var act = outs[gi].Data;
        if (mode == "scores6" && exp.Length % 6 == 0 && act.Length == exp.Length)
        {
            exp = Enumerable.Range(0, exp.Length / 6).Select(q => exp[q * 6 + 4]).ToArray();
            act = Enumerable.Range(0, act.Length / 6).Select(q => act[q * 6 + 4]).ToArray();
        }
        if (exp.Length != act.Length) { cmps.Add(new { name = outNames[k], error = $"size {act.Length} vs ref {exp.Length}" }); allOk = false; continue; }
        var c = Check(act, exp, rtol, atol);
        bool ok = rmsTol is { } t ? !double.IsNaN(c.MaxAbs) && c.Rms <= t : c.Ok;
        allOk &= ok; worstRms = Math.Max(worstRms, double.IsNaN(c.Rms) ? 1e9 : c.Rms);
        cmps.Add(new { name = outNames[k], n = exp.Length, maxAbs = c.MaxAbs, maxRel = c.MaxRel, rms = c.Rms, scale = c.Scale, ok });
    }
    res["outputs"] = cmps; res["parityOk"] = allOk; res["worstRms"] = worstRms;
    // timing: Run + read back every output (forces completion), min over runs
    var times = new List<double>(); var total = Stopwatch.StartNew();
    var buf = graph.Outputs.ToDictionary(o => o.Key, o => new byte[(int)o.Value.Type.ByteSize(o.Value.Shape.Elements(run.Symbols))]);
    while (times.Count < maxRuns && (times.Count < 3 || total.Elapsed.TotalSeconds < budgetS))
    {
        sw.Restart();
        compiled.Run(run);
        foreach (var (on, b) in buf) compiled.GetOutput(on, b);
        times.Add(sw.Elapsed.TotalMilliseconds);
    }
    res["minMs"] = Math.Round(times.Min(), 3); res["medMs"] = Math.Round(times.OrderBy(x => x).ElementAt(times.Count / 2), 3); res["runs"] = times.Count;
    if (argv.Contains("--profile") && providerName == "vulkan")
    {
        foreach (var seg in compiled.Segments.OfType<VulkanSegment>()) { seg.Profile = true; seg.ResetProfile(); }
        for (int q = 0; q < 3; q++) compiled.Run(run);
        res["profile"] = string.Join("\n", compiled.Segments.OfType<VulkanSegment>().Select(sg => sg.ProfileReport(12)));
    }
    res["stage"] = "ok";
    def?.OwnedResources.ToList().ForEach(d => d.Dispose());
    Emit(); return 0;
}
catch (Exception e)
{
    res["error"] = Trunc(e.GetType().Name + ": " + e.Message, 1500);
    Emit(); return 1;
}

static string Trunc(string? s, int n = 600) => s is null ? "" : (s.Length > n ? s[..n] + "..." : s);
static float[] ReadF32(string p) { var b = File.ReadAllBytes(p); var f = new float[b.Length / 4]; Buffer.BlockCopy(b, 0, f, 0, b.Length); return f; }

static byte[] FromF32(float[] f, DataType t)
{
    if (t == DataType.F32) { var b = new byte[f.Length * 4]; Buffer.BlockCopy(f, 0, b, 0, b.Length); return b; }
    if (t == DataType.I32) { var a = f.Select(x => (int)x).ToArray(); var b = new byte[a.Length * 4]; Buffer.BlockCopy(a, 0, b, 0, b.Length); return b; }
    if (t == DataType.I64) { var a = f.Select(x => (long)x).ToArray(); var b = new byte[a.Length * 8]; Buffer.BlockCopy(a, 0, b, 0, b.Length); return b; }
    if (t == DataType.F16) { var a = f.Select(x => (Half)x).ToArray(); var b = new byte[a.Length * 2]; Buffer.BlockCopy(a, 0, b, 0, b.Length); return b; }
    if (t == DataType.U8 || t == DataType.Bool) return f.Select(x => (byte)x).ToArray();
    if (t == DataType.I8) return f.Select(x => (byte)(sbyte)x).ToArray();
    if (t == DataType.F64) { var a = f.Select(x => (double)x).ToArray(); var b = new byte[a.Length * 8]; Buffer.BlockCopy(a, 0, b, 0, b.Length); return b; }
    throw new NotSupportedException($"harness cannot feed input type {t}");
}

static float[] ReadOut(CompiledModel c, string name, Value v, RunParameters run)
{
    long n = v.Shape.Elements(run.Symbols); var raw = new byte[(int)v.Type.ByteSize(n)];
    c.GetOutput(name, raw);
    var o = new float[n];
    if (v.Type == DataType.F32) Buffer.BlockCopy(raw, 0, o, 0, raw.Length);
    else if (v.Type == DataType.I32) for (int i = 0; i < n; i++) o[i] = BitConverter.ToInt32(raw, i * 4);
    else if (v.Type == DataType.I64) for (int i = 0; i < n; i++) o[i] = BitConverter.ToInt64(raw, i * 8);
    else if (v.Type == DataType.F16) for (int i = 0; i < n; i++) o[i] = (float)BitConverter.ToHalf(raw, i * 2);
    else if (v.Type == DataType.F64) for (int i = 0; i < n; i++) o[i] = (float)BitConverter.ToDouble(raw, i * 8);
    else if (v.Type == DataType.U8 || v.Type == DataType.Bool) for (int i = 0; i < n; i++) o[i] = raw[i];
    else if (v.Type == DataType.I8) for (int i = 0; i < n; i++) o[i] = (sbyte)raw[i];
    else throw new NotSupportedException($"harness cannot read output type {v.Type}");
    return o;
}

static (double MaxAbs, double MaxRel, double Scale, double Rms, bool Ok) Check(float[] a, float[] e, double rtol, double atolScale)
{
    double scale = Math.Max(1.0, e.Select(Math.Abs).DefaultIfEmpty(0).Max());
    double atol = atolScale * scale, maxAbs = 0, maxRel = 0, se = 0, sd = 0; bool ok = true;
    for (int i = 0; i < e.Length; i++)
    {
        if (float.IsNaN(a[i]) || float.IsInfinity(a[i])) { if (float.IsNaN(e[i]) || float.IsInfinity(e[i]) && a[i] == e[i]) continue; return (double.NaN, double.NaN, scale, double.NaN, false); }
        if (float.IsNaN(e[i]) || float.IsInfinity(e[i])) continue;
        double d = Math.Abs((double)a[i] - e[i]); se += (double)e[i] * e[i]; sd += d * d;
        maxAbs = Math.Max(maxAbs, d); maxRel = Math.Max(maxRel, d / Math.Max(Math.Abs(e[i]), 1e-2 * scale));
        if (d > atol + rtol * Math.Abs(e[i])) ok = false;
    }
    return (maxAbs, maxRel, scale, Math.Sqrt(sd / Math.Max(se, 1e-30)), ok);
}
