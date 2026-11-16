// Parity harness of the top-20 sweep.
//
//   ParitySweep <model path> <spec.json> <out.json> [--cpu] [--gen N] [--no-gen] [--teacher reference.json] [--convert package.slm] [--devices N] [--cpu-offload]
//
// spec.json: { "texts": [..], "chats": [{ "name", "messages": [..], "tools": [..], "context": {..} }], "prompts": [..], "chat_prompts": [names] }
// Output: tokenizer ids + decoded text per text, rendered chat text + ids per conversation, greedy ids per prompt.
using System.Text.Json;
using System.Text.Json.Nodes;
using Sharplight.Llm;
using Sharplight.Runtime;
using Sharplight.Samples;
using Sharplight.Templating;

if (args.Length < 3)
{
    Console.Error.WriteLine("usage: ParitySweep <model> <spec.json> <out.json> [--cpu] [--gen N] [--no-gen]");
    return 2;
}

string modelPath = args[0];
var spec = JsonNode.Parse(File.ReadAllText(args[1]))!.AsObject();
string outPath = args[2];
bool cpu = args.Contains("--cpu");
bool noGen = args.Contains("--no-gen");
string? teacher = null;
string? convert = null;
int deviceCount = 1;
bool cpuOffload = args.Contains("--cpu-offload");
int gen = 48;
for (int i = 3; i < args.Length - 1; i++)
{
    if (args[i] == "--gen")
    {
        gen = int.Parse(args[i + 1]);
    }
    else if (args[i] == "--devices")
    {
        deviceCount = int.Parse(args[i + 1]);
    }
    else if (args[i] == "--convert")
    {
        convert = args[i + 1];
    }
    else if (args[i] == "--teacher")
    {
        teacher = args[i + 1];
    }
}

using var runtime = new SampleRuntime();
var provider = runtime.SelectProvider(cpu);
// --devices N: the model is placed over the first N devices of the provider (equal weights); --cpu-offload adds the CPU for what does not fit.
var slots = Enumerable.Range(0, deviceCount)
    .Select(index => new DeviceSlot { Device = provider.OpenDevice(SampleRuntime.GetDevice(provider, index), new DeviceOptions()), Weight = 1 })
    .ToList();
DeviceSlot? fallback = null;
if (cpuOffload)
{
    var cpuProvider = runtime.SelectProvider(cpu: true);
    fallback = new DeviceSlot { Device = cpuProvider.OpenDevice(SampleRuntime.GetDevice(cpuProvider, 0), new DeviceOptions()) };
}

var result = new JsonObject { ["model"] = Path.GetFileName(modelPath.TrimEnd('/')), ["device"] = cpu ? "cpu" : "vulkan" };
var sw = System.Diagnostics.Stopwatch.StartNew();
if (convert is not null)
{
    // Convert to a .slm package first and run every check on the package instead of the original files.
    runtime.Converter.Convert(modelPath, convert);
    result["convert_ms"] = sw.ElapsedMilliseconds;
    result["package_bytes"] = new FileInfo(convert).Length;
    modelPath = convert;
    sw.Restart();
}

using var model = LlmModel.Load(runtime.Loader, modelPath, new LlmLoadOptions
{
    Devices = slots,
    Fallback = fallback,
    ContextLength = 2048,
});
result["load_ms"] = sw.ElapsedMilliseconds;
var tok = model.Tokenizer;

var texts = new JsonArray();
foreach (var node in spec["texts"]?.AsArray() ?? [])
{
    string text = node!.GetValue<string>();
    var entry = new JsonObject();
    try
    {
        int[] ids = tok.Encode(text, addSpecial: false, parseSpecial: false);
        entry["ids"] = new JsonArray([.. ids.Select(id => (JsonNode)id)]);
        entry["decoded"] = tok.Decode(ids);
    }
    catch (Exception error)
    {
        entry["error"] = error.GetType().Name + ": " + error.Message;
    }

    texts.Add(entry);
}

result["texts"] = texts;

string? bos = tok.BosId is int b ? tok.TokenString(b) : null;
string? eos = tok.EosId is int e ? tok.TokenString(e) : null;
var template = model.ChatTemplate is { } src ? new ChatTemplate(src, bos, eos) : null;
result["has_template"] = template is not null;
var chats = new JsonObject();
var chatIds = new Dictionary<string, int[]>(StringComparer.Ordinal);
foreach (var node in spec["chats"]?.AsArray() ?? [])
{
    var chat = node!.AsObject();
    string name = chat["name"]!.GetValue<string>();
    var entry = new JsonObject();
    if (template is null)
    {
        entry["error"] = "no template";
        chats[name] = entry;
        continue;
    }

    try
    {
        var messages = chat["messages"]!.AsArray().Select(m => ToMessage(m!.AsObject())).ToList();
        var tools = chat["tools"]?.AsArray().Select(t => ToolDefinition.FromJson(t!.ToJsonString())).ToList();
        Dictionary<string, object?>? context = null;
        if (chat["context"] is JsonObject ctx)
        {
            context = ctx.ToDictionary(kv => kv.Key, kv => ToClr(kv.Value));
        }

        bool addGen = chat["add_generation_prompt"]?.GetValue<bool>() ?? true;
        string text = template.Render(messages, addGen, context, tools);
        // As ModelService: BOS added when the tokenizer adds it and the template did not.
        bool addSpecial = tok.AddBos && bos is not null && !text.StartsWith(bos, StringComparison.Ordinal);
        int[] ids = tok.Encode(text, addSpecial, parseSpecial: true);
        entry["text"] = text;
        entry["ids"] = new JsonArray([.. ids.Select(id => (JsonNode)id)]);
        chatIds[name] = ids;
    }
    catch (Exception error)
    {
        entry["error"] = error.GetType().Name + ": " + error.Message;
    }

    chats[name] = entry;
}

result["chats"] = chats;

if (!noGen)
{
    var greedy = new JsonArray();
    using var session = model.CreateSession();
    var generator = new Generator(session, tok, null);
    var options = new GenerationOptions { MaxNewTokens = gen, Sampling = SamplingOptions.Greedy, StopAtEog = false };
    var prompts = new List<(string Name, int[] Ids)>();
    foreach (var node in spec["prompts"]?.AsArray() ?? [])
    {
        string prompt = node!.GetValue<string>();
        prompts.Add((prompt, tok.Encode(prompt, addSpecial: true, parseSpecial: false)));
    }

    foreach (var node in spec["chat_prompts"]?.AsArray() ?? [])
    {
        string name = node!.GetValue<string>();
        if (chatIds.TryGetValue(name, out var ids))
        {
            prompts.Add(("chat:" + name, ids));
        }
    }

    foreach (var (name, ids) in prompts)
    {
        var entry = new JsonObject { ["prompt"] = name, ["prompt_ids"] = new JsonArray([.. ids.Select(id => (JsonNode)id)]) };
        try
        {
            session.Reset();
            var outIds = generator.Generate(ids, options).Select(t => t.Id).ToArray();
            entry["ids"] = new JsonArray([.. outIds.Select(id => (JsonNode)id)]);
            entry["text"] = tok.Decode(outIds);
        }
        catch (Exception error)
        {
            entry["error"] = error.GetType().Name + ": " + error.Message;
        }

        greedy.Add(entry);
    }

    result["greedy"] = greedy;

    // Teacher forcing on the reference's tokens: our logits at the reference's top-5 ids per step (precision diagnostics).
    if (teacher is not null)
    {
        var forced = new JsonArray();
        foreach (var node in JsonNode.Parse(File.ReadAllText(teacher))!["greedy"]!.AsArray())
        {
            if (node!["top5"] is not JsonArray top5)
            {
                continue;
            }

            session.Reset();
            int[] promptIds = [.. node["prompt_ids"]!.AsArray().Select(x => x!.GetValue<int>())];
            int[] refIds = [.. node["ids"]!.AsArray().Select(x => x!.GetValue<int>())];
            float[] logits = session.Process(promptIds);
            var steps = new JsonArray();
            for (int step = 0; step < refIds.Length; step++)
            {
                var row = new JsonArray();
                foreach (var pair in top5[step]!.AsArray())
                {
                    int id = pair![0]!.GetValue<int>();
                    row.Add(new JsonArray(id, Math.Round(logits[id], 4)));
                }

                int best = Array.IndexOf(logits, logits.Max());
                steps.Add(new JsonObject { ["at_ref"] = row, ["argmax"] = best, ["max"] = Math.Round(logits[best], 4) });
                logits = session.Process([refIds[step]]);
            }

            forced.Add(new JsonObject { ["prompt"] = node["prompt"]!.GetValue<string>(), ["steps"] = steps });
        }

        result["teacher"] = forced;
    }
}

File.WriteAllText(outPath, result.ToJsonString(new JsonSerializerOptions { WriteIndented = true, Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping }));
foreach (var slot in slots)
{
    slot.Device.Dispose();
}

fallback?.Device.Dispose();
return 0;

static ChatMessage ToMessage(JsonObject m)
{
    string role = m["role"]!.GetValue<string>();
    string? content = m["content"]?.GetValue<string>();
    string? reasoning = m["reasoning_content"]?.GetValue<string>();
    List<ChatToolCall>? calls = null;
    if (m["tool_calls"] is JsonArray tc)
    {
        calls = [];
        foreach (var c in tc)
        {
            var fn = c!["function"]!.AsObject();
            var arguments = fn["arguments"] is JsonObject a ? a.ToDictionary(kv => kv.Key, kv => ToClr(kv.Value)) : null;
            calls.Add(new ChatToolCall(c["id"]?.GetValue<string>(), fn["name"]!.GetValue<string>(), arguments));
        }
    }

    return new ChatMessage(role, content, null, reasoning, calls, m["name"]?.GetValue<string>(), m["tool_call_id"]?.GetValue<string>());
}

static object? ToClr(JsonNode? node)
{
    return node switch
    {
        null => null,
        JsonObject o => o.ToDictionary(kv => kv.Key, kv => ToClr(kv.Value)),
        JsonArray a => a.Select(ToClr).ToList(),
        JsonValue v when v.TryGetValue(out bool flag) => flag,
        JsonValue v when v.TryGetValue(out long n) => n,
        JsonValue v when v.TryGetValue(out double d) => d,
        JsonValue v => v.GetValue<string>(),
    };
}
