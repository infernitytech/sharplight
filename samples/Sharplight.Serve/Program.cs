// sharplight-serve: load one or several models and serve them over the OpenAI and/or Anthropic HTTP API.
//
//   sharplight-serve <model.gguf | model.slm | hf-checkpoint-dir> [options]            one model (positional)
//   sharplight-serve --model NAME=PATH[,key=value...] [--model ...] [options]          several models
//   sharplight-serve --config models.json [options]                                     several models from a file
//
//   --listen http://host:port   address to listen on (alias --urls; default http://127.0.0.1:8080)
//   --api openai|anthropic|both which API to expose (default both)
//   --api-key KEY               require this key (Authorization: Bearer / x-api-key); repeatable. Also protects /admin.
//   --model-name NAME           model id reported by the API for the positional model (default: file/directory name)
//   --device N | --cpu          GPU device index (default 0) / run on the CPU
//   --ctx N --batch N           context length per sequence (4096) / rows per forward pass (512)
//   --slots N                   concurrent sequences batched per forward pass, each with --ctx context (default 4)
//   --max-queue N               waiting requests before answering 429 (default 64)
//   --prefill-chunk N           prompt rows per step and sequence, interleaved with decoding (default 512)
//   --cpu-offload --disable-features a,b   as in the CLI
//   --mmproj vision.gguf        vision projector (images are not served yet)
//   --cors ORIGIN               allow a CORS origin ("*" for any); repeatable
//   --chat-template FILE        use this Jinja chat template instead of the model's
//   --temp T                    default temperature for requests that give none (default 0.7)
//
// Several models share each opened device (one Vulkan device per GPU) and run concurrently, each with its own scheduler.
// The request's "model" field selects the model; GET /v1/models lists all of them with their state.
//
//   --model NAME=PATH,ctx=N,slots=N,batch=N,device=N|cpu,mmproj=FILE,template=FILE,lazy=true|false,pin=true,idle=SECONDS,offload=true
//                               per-model settings; anything left out takes the global option above
//   --config FILE               JSON: {"vramReserve":"2G","maxLoaded":2,"idleTimeout":600,"lazy":true,
//                                      "models":[{"name":"qwen","path":"...","ctx":4096,"slots":4,"device":0,"lazy":false,"pin":false,"offload":true}]}
//   --vram-reserve SIZE         keep this much device memory free for other applications (default 1G; K/M/G suffixes)
//   --vram-budget SIZE          never use more than this much device memory in total (default: what the driver reports as available)
//   --max-loaded N              at most N models resident at once; loading another evicts the least recently used idle one
//   --lazy                      load models on their first request instead of at startup (default: at startup)
//   --idle-timeout SECONDS      unload models that served nothing for this long (they reload on demand)
//
// Models that do not fit in the free device memory are refused with a clear error (or, with --cpu-offload, partly run on the CPU);
// idle models are evicted (least recently used first) to make room for the one requested.
// Admin (needs the API key, or a local caller when no key is set): GET /admin/models, POST /admin/models/NAME/load|unload[?force=true]
//
// Examples:
//   dotnet run -c Release --project samples/Sharplight.Serve -- ~/models/gguf/Qwen3-0.6B/Qwen3-0.6B-Q8_0.gguf --listen http://0.0.0.0:8080
//   dotnet run -c Release --project samples/Sharplight.Serve -- --model qwen=~/models/gguf/Qwen3-0.6B/Qwen3-0.6B-Q8_0.gguf,slots=4 \
//        --model smol=~/models/gguf/SmolLM2-360M-Instruct/smollm2-360m-instruct-q8_0.gguf,ctx=2048 --vram-reserve 4G --lazy
//   curl localhost:8080/v1/chat/completions -H 'content-type: application/json' \
//        -d '{"model":"qwen","messages":[{"role":"user","content":"Say hi"}],"max_tokens":32}'
//   curl -N localhost:8080/v1/chat/completions -H 'content-type: application/json' \
//        -d '{"model":"smol","stream":true,"messages":[{"role":"user","content":"Count to 5"}]}'
//   curl localhost:8080/v1/messages -H 'content-type: application/json' -H 'x-api-key: KEY' -H 'anthropic-version: 2023-06-01' \
//        -d '{"model":"qwen","max_tokens":64,"messages":[{"role":"user","content":"Say hi"}]}'
//   curl localhost:8080/v1/models ; curl localhost:8080/admin/models ; curl -X POST localhost:8080/admin/models/smol/unload
//   curl localhost:8080/health ; curl localhost:8080/metrics
//
// Python: openai.OpenAI(base_url="http://localhost:8080/v1", api_key="x") / anthropic.Anthropic(base_url="http://localhost:8080", api_key="x")

using System.Globalization;
using System.Text.Json;
using Sharplight.Cpu;
using Sharplight.Gguf;
using Sharplight.HuggingFace;
using Sharplight.Llm;
using Sharplight.Llm.Serving;
using Sharplight.Llm.Serving.Chat;
using Sharplight.Llm.Serving.Hosting;
using Sharplight.Package;
using Sharplight.Runtime;
using Sharplight.Server;
using Sharplight.Vulkan;

var argv = args.ToList();
string? Opt(string name, string? def = null) { int i = argv.IndexOf(name); return i >= 0 && i + 1 < argv.Count ? argv[i + 1] : def; }
List<string> Opts(string name) { var l = new List<string>(); for (int i = 0; i + 1 < argv.Count; i++) { if (argv[i] == name) { l.Add(argv[i + 1]); } } return l; }
bool Flag(string name) => argv.Contains(name);

var modelArgs = Opts("--model");
var configPath = Opt("--config");
bool positional = argv.Count > 0 && !argv[0].StartsWith("--");
if (!positional && modelArgs.Count == 0 && configPath is null)
{
    Console.Error.WriteLine("usage: sharplight-serve <model.gguf | model.slm | hf-dir> | --model NAME=PATH[,key=value...] ... | --config models.json  [--listen url] [--api openai|anthropic|both] [--api-key k] [--device n] [--ctx n] [--batch n] [--slots n] [--vram-reserve 2G] [--max-loaded n] [--lazy] [--idle-timeout s] ...");
    return 1;
}

ProviderRegistry.Register(() => new VulkanProvider());
CpuExecutionProvider.Register();
GgufImporter.Register();
HuggingFaceImporter.Register();
ModelPackageImporter.Register();

// --- model specs: positional model, --model NAME=PATH[,k=v...], and/or a JSON config; global options are the defaults ---
var specs = new List<Spec>();
Config? config = configPath is null ? null : JsonSerializer.Deserialize<Config>(File.ReadAllText(configPath), new JsonSerializerOptions { PropertyNameCaseInsensitive = true, ReadCommentHandling = JsonCommentHandling.Skip, AllowTrailingCommas = true })
    ?? throw new InvalidOperationException("Empty config file.");
if (positional)
{
    specs.Add(new Spec { Name = Opt("--model-name") ?? Path.GetFileNameWithoutExtension(argv[0].TrimEnd('/', '\\')), Path = argv[0] });
}

foreach (var m in modelArgs)
{
    specs.Add(ParseSpec(m));
}

if (config?.Models is { } cm)
{
    specs.AddRange(cm);
}

if (specs.Select(s => s.Name).Distinct().Count() != specs.Count) { Console.Error.WriteLine("Model names must be unique."); return 1; }

long reserve = ParseSize(Opt("--vram-reserve") ?? config?.VramReserve ?? "1G");
long? budget = (Opt("--vram-budget") ?? config?.VramBudget) is { } vb ? ParseSize(vb) : null;
int maxLoaded = int.Parse(Opt("--max-loaded") ?? config?.MaxLoaded?.ToString() ?? "0");
bool lazyAll = Flag("--lazy") || config?.Lazy == true;
double? idleSeconds = Opt("--idle-timeout") is { } its ? double.Parse(its, CultureInfo.InvariantCulture) : config?.IdleTimeout;

// --- devices: one opened device per physical GPU, shared by all models ---
using var devices = new DevicePool();
var devOptions = new DeviceOptions { MemoryBudget = budget };
foreach (var f in (Opt("--disable-features") ?? "").Split(',', StringSplitOptions.RemoveEmptyEntries))
{
    devOptions.DisabledFeatures.Add(f.Trim());
}

IComputeDevice OpenDevice(string spec)
{
    bool cpu = spec.Equals("cpu", StringComparison.OrdinalIgnoreCase);
    var provider = ProviderRegistry.Discover().First(x => x.Availability.IsAvailable && (x.Provider.Name == "cpu") == cpu).Provider;
    return devices.Open(provider, provider.GetDevices()[cpu ? 0 : int.Parse(spec)], cpu ? new DeviceOptions() : devOptions);
}
IComputeDevice? cpuDevice = null;
IComputeDevice Cpu() => cpuDevice ??= OpenDevice("cpu");

// --- host: registry with lazy/eager loading, eviction and the memory reserve ---
var host = new ModelHost(new ModelHostOptions
{
    ReserveBytes = reserve,
    MaxLoadedModels = maxLoaded,
    IdleTimeout = idleSeconds is > 0 ? TimeSpan.FromSeconds(idleSeconds.Value) : null,
    Log = Console.Error.WriteLine,
});
foreach (var s in specs)
{
    string dev = s.Device ?? (Flag("--cpu") ? "cpu" : Opt("--device", "0")!);
    bool onCpu = dev.Equals("cpu", StringComparison.OrdinalIgnoreCase);
    DeviceSlot? fallback = !onCpu && (s.Offload ?? Flag("--cpu-offload")) ? new DeviceSlot { Device = Cpu() } : null;
    var mm = s.Mmproj ?? Opt("--mmproj");
    var load = new LlmLoadOptions
    {
        Auxiliary = mm is not null ? [("mmproj", mm)] : [],
        Devices = [new DeviceSlot { Device = OpenDevice(dev) }],
        Fallback = fallback,
        ContextLength = s.Ctx ?? int.Parse(Opt("--ctx", "4096")!),
        MaxBatchTokens = s.Batch ?? int.Parse(Opt("--batch", "512")!),
        Log = lazyAll || specs.Count > 1 ? m => Console.Error.WriteLine($"[{s.Name}] {m}") : Console.Error.WriteLine,
    };
    var tpl = s.Template ?? Opt("--chat-template");
    var chat = new ChatModelOptions
    {
        Slots = s.Slots ?? int.Parse(Opt("--slots", "4")!),
        Engine = new EngineOptions { MaxQueue = int.Parse(Opt("--max-queue", "64")!), PrefillChunk = int.Parse(Opt("--prefill-chunk", "512")!) },
        Service = new ModelServiceOptions { ChatTemplate = tpl is not null ? File.ReadAllText(tpl) : null },
    };
    host.Register(new ModelRegistration
    {
        Name = s.Name,
        Loader = new LlmChatModelLoader(s.Name, s.Path, load, chat),
        LoadAtStartup = !(s.Lazy ?? lazyAll),
        Pinned = s.Pin ?? false,
        IdleTimeout = s.Idle is > 0 ? TimeSpan.FromSeconds(s.Idle.Value) : null,
        Description = s.Path,
    });
}
try { await host.LoadStartupModelsAsync(); }
catch (Exception e)
{
    Console.Error.WriteLine("Cannot load model: " + e.Message);
    host.Dispose();
    return 2;
}

// --- HTTP ---
string api = Opt("--api", "both")!;
var builder = WebApplication.CreateBuilder();
builder.WebHost.UseUrls(Opt("--listen") ?? Opt("--urls") ?? "http://127.0.0.1:8080");
builder.WebHost.ConfigureKestrel(k => k.Limits.MaxRequestBodySize = null);
builder.Logging.SetMinimumLevel(LogLevel.Warning);
builder.Services.AddSharplight(host, o =>
{
    o.OpenAi = api is "openai" or "both";
    o.Anthropic = api is "anthropic" or "both";
    o.ApiKeys.AddRange(Opts("--api-key"));
    o.CorsOrigins.AddRange(Opts("--cors"));
    o.DefaultSampling = new SamplingParams { Temperature = float.Parse(Opt("--temp", "0.7")!, CultureInfo.InvariantCulture) };
    // A single model answers to any name (clients often send arbitrary ones); with several, an unknown name is a 404.
    o.UnknownModelFallsBackToDefault = specs.Count == 1;
});
var app = builder.Build();
app.UseSharplight();
app.MapSharplight();
Console.Error.WriteLine($"Serving {string.Join(", ", specs.Select(s => $"'{s.Name}'"))} ({api} API, VRAM reserve {reserve >> 20} MiB{(maxLoaded > 0 ? $", max {maxLoaded} loaded" : "")}) on {string.Join(", ", builder.WebHost.GetSetting("urls")?.Split(';') ?? [])}");
Console.Error.WriteLine(host.Describe());
try { await app.RunAsync(); }
finally { host.Dispose(); }   // engines first, then (below) the devices they run on
return 0;

// ---------------------------------------------------------------------------------------------------------------------------

static long ParseSize(string s)
{
    s = s.Trim();
    long mult = 1;
    char last = char.ToUpperInvariant(s[^1]);
    if (last is 'K' or 'M' or 'G' or 'T') { mult = 1L << (last switch { 'K' => 10, 'M' => 20, 'G' => 30, _ => 40 }); s = s[..^1].TrimEnd('i', 'I', 'b', 'B'); }
    else if (last == 'B')
    {
        s = s[..^1];
    }

    return (long)(double.Parse(s, CultureInfo.InvariantCulture) * mult);
}

static Spec ParseSpec(string text)
{
    int eq = text.IndexOf('=');
    if (eq <= 0)
    {
        throw new ArgumentException($"--model expects NAME=PATH[,key=value...], got '{text}'.");
    }

    var parts = text[(eq + 1)..].Split(',');
    var spec = new Spec { Name = text[..eq], Path = parts[0] };
    foreach (var kv in parts.Skip(1))
    {
        var p = kv.Split('=', 2);
        string v = p.Length > 1 ? p[1] : "true";
        switch (p[0].ToLowerInvariant())
        {
            case "ctx": spec.Ctx = int.Parse(v); break;
            case "slots": spec.Slots = int.Parse(v); break;
            case "batch": spec.Batch = int.Parse(v); break;
            case "device": spec.Device = v; break;
            case "mmproj": spec.Mmproj = v; break;
            case "template": spec.Template = v; break;
            case "lazy": spec.Lazy = bool.Parse(v); break;
            case "pin": spec.Pin = bool.Parse(v); break;
            case "offload": spec.Offload = bool.Parse(v); break;
            case "idle": spec.Idle = double.Parse(v, CultureInfo.InvariantCulture); break;
            default: throw new ArgumentException($"Unknown model setting '{p[0]}' in --model {text}.");
        }
    }
    return spec;
}

internal sealed class Spec
{
    public string Name { get; set; } = "";
    public string Path { get; set; } = "";
    public int? Ctx { get; set; }
    public int? Slots { get; set; }
    public int? Batch { get; set; }
    public string? Device { get; set; }
    public string? Mmproj { get; set; }
    public string? Template { get; set; }
    public bool? Lazy { get; set; }
    public bool? Pin { get; set; }
    public bool? Offload { get; set; }
    public double? Idle { get; set; }
}

internal sealed class Config
{
    public string? VramReserve { get; set; }
    public string? VramBudget { get; set; }
    public int? MaxLoaded { get; set; }
    public bool? Lazy { get; set; }
    public double? IdleTimeout { get; set; }
    public List<Spec>? Models { get; set; }
}
