using System.Globalization;
using System.Text;
using System.Text.Json;
using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>
/// Translates a Hugging Face tokenizer (<c>tokenizer.json</c>, <c>tokenizer_config.json</c>, <c>config.json</c>,
/// chat template files) into the canonical <c>tokenizer.*</c> metadata, following llama.cpp's
/// <c>TextModel._set_vocab_gpt2</c> / <c>get_vocab_base</c> and gguf-py's <c>SpecialVocab</c>.
/// Only byte-level BPE ("gpt2") tokenizers are supported so far.
/// </summary>
public static class HfTokenizerTranslator
{
    const int TypeNormal = 1, TypeControl = 3, TypeUserDefined = 4, TypeUnused = 5;

    /// <summary>Known pre-tokenizer split regexes (as found in tokenizer.json) → Sharplight/llama.cpp preset names.</summary>
    static readonly (string Preset, string Regex)[] KnownRegexes =
    [
        ("qwen35", @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"),
        ("qwen2", @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"),
        ("llama-bpe", @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"),
        ("gpt-2", @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"),
    ];

    public static void AddTo(IDictionary<string, object> md, string dir, HfConfig config)
    {
        string tokPath = Path.Combine(dir, "tokenizer.json");
        if (!File.Exists(tokPath))
            throw new NotSupportedException($"{dir}: no tokenizer.json (SentencePiece-only tokenizers are not supported yet).");
        using var tokDoc = JsonDocument.Parse(File.ReadAllBytes(tokPath));
        var tok = tokDoc.RootElement;

        var model = tok.GetProperty("model");
        string modelType = model.TryGetProperty("type", out var mt) ? mt.GetString() ?? "" : "";
        if (modelType != "BPE") throw new NotSupportedException($"tokenizer.json model type '{modelType}' is not supported (only BPE).");

        JsonElement? tcRoot = null;
        using var tcDoc = ReadOptional(Path.Combine(dir, "tokenizer_config.json"));
        if (tcDoc != null) tcRoot = tcDoc.RootElement;

        var addedTokens = tok.TryGetProperty("added_tokens", out var at) && at.ValueKind == JsonValueKind.Array
            ? at.EnumerateArray().ToArray() : [];

        // ---- tokens + types (llama.cpp get_vocab_base) ---------------------------------------
        var vocab = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var p in model.GetProperty("vocab").EnumerateObject()) vocab[p.Name] = p.Value.GetInt32();
        var addedById = new Dictionary<int, JsonElement>();
        foreach (var a in addedTokens)
        {
            vocab[a.GetProperty("content").GetString()!] = a.GetProperty("id").GetInt32(); // transformers' vocab includes added tokens
            addedById[a.GetProperty("id").GetInt32()] = a;
        }
        // transformers also merges tokenizer_config.json's added_tokens_decoder (it may list tokens tokenizer.json lacks).
        if (tcRoot is { ValueKind: JsonValueKind.Object } tcr && tcr.TryGetProperty("added_tokens_decoder", out var atd) && atd.ValueKind == JsonValueKind.Object)
            foreach (var e in atd.EnumerateObject())
            {
                if (!int.TryParse(e.Name, NumberStyles.None, CultureInfo.InvariantCulture, out int id) || addedById.ContainsKey(id)) continue;
                if (e.Value.TryGetProperty("content", out var content) && content.GetString() is { } cs)
                {
                    vocab[cs] = id;
                    addedById[id] = e.Value;
                }
            }
        int maxId = vocab.Count == 0 ? -1 : vocab.Values.Max();
        int vocabSize = config.GetInt("vocab_size") ?? vocab.Count;
        // Tokens beyond the model's embedding table cannot be produced or fed (e.g. Gemma 3's extra added token): drop them.
        if (maxId >= vocabSize)
            foreach (var id in vocab.Where(kv => kv.Value >= vocabSize).Select(kv => kv.Key).ToList()) vocab.Remove(id);

        var reverse = new Dictionary<int, string>(vocab.Count);
        foreach (var (t, id) in vocab) reverse[id] = t;
        var tokens = new string[vocabSize];
        var types = new int[vocabSize];
        var addedContents = new HashSet<string>(addedById.Values.Select(a => a.GetProperty("content").GetString()!), StringComparer.Ordinal);
        for (int i = 0; i < vocabSize; i++)
        {
            if (!reverse.TryGetValue(i, out var token)) { tokens[i] = $"[PAD{i}]"; types[i] = TypeUnused; continue; }
            if (addedContents.Contains(token) && addedById.TryGetValue(i, out var a))
            {
                // TODO: llama.cpp re-normalizes non-"normalized" added tokens via decode(encode(tok)); identity for common vocabularies.
                bool special = a.TryGetProperty("special", out var sp) && sp.ValueKind == JsonValueKind.True;
                if (special || LooksSpecial(token)) types[i] = TypeControl;
                else { token = token.Replace("▁", " "); types[i] = TypeUserDefined; }
            }
            else types[i] = TypeNormal;
            tokens[i] = token;
        }

        md[CanonicalKeys.TokenizerModel] = "gpt2";
        md[CanonicalKeys.TokenizerPre] = DetectPreTokenizer(tok);
        md[CanonicalKeys.TokenizerTokens] = tokens;
        md[CanonicalKeys.TokenizerTokenTypes] = types;
        if (ReadMerges(model) is { Length: > 0 } merges) md[CanonicalKeys.TokenizerMerges] = merges;

        AddSpecialVocab(md, dir, config, tok, tcRoot, addedTokens);
    }

    static bool LooksSpecial(string t)
        => t is "<pad>" or "<mask>" or "<2mass>" or "[@BOS@]"
           || (t.StartsWith("<|", StringComparison.Ordinal) && t.EndsWith("|>", StringComparison.Ordinal))
           || (t.StartsWith("<｜", StringComparison.Ordinal) && t.EndsWith("｜>", StringComparison.Ordinal))
           || (t.StartsWith("<unused", StringComparison.Ordinal) && t.EndsWith('>'));

    // ---- pre-tokenizer ------------------------------------------------------------------------
    static string DetectPreTokenizer(JsonElement tok)
    {
        var regexes = new List<string>();
        bool byteLevelRegex = false;
        if (tok.TryGetProperty("pre_tokenizer", out var pre) && pre.ValueKind == JsonValueKind.Object) Collect(pre);

        void Collect(JsonElement p)
        {
            string type = p.TryGetProperty("type", out var t) ? t.GetString() ?? "" : "";
            if (type == "Sequence" && p.TryGetProperty("pretokenizers", out var arr))
                foreach (var e in arr.EnumerateArray()) Collect(e);
            else if (type == "Split" && p.TryGetProperty("pattern", out var pat) && pat.TryGetProperty("Regex", out var rx))
                regexes.Add(rx.GetString() ?? "");
            else if (type == "ByteLevel" && (!p.TryGetProperty("use_regex", out var ur) || ur.ValueKind != JsonValueKind.False))
                byteLevelRegex = true;
        }

        if (byteLevelRegex && regexes.Count == 0) return "gpt-2";
        foreach (var r in regexes)
            foreach (var (preset, known) in KnownRegexes)
                if (r == known) return preset;
        throw new NotSupportedException(
            "Unknown BPE pre-tokenizer; add its split regex to HfTokenizerTranslator.KnownRegexes. Regexes found: " +
            (regexes.Count == 0 ? "(none)" : string.Join(" || ", regexes)));
    }

    // ---- merges ---------------------------------------------------------------------------------
    static string[]? ReadMerges(JsonElement model)
    {
        if (!model.TryGetProperty("merges", out var m) || m.ValueKind != JsonValueKind.Array) return null;
        var list = new string[m.GetArrayLength()];
        int i = 0;
        foreach (var e in m.EnumerateArray())
        {
            if (e.ValueKind == JsonValueKind.String) list[i++] = e.GetString()!;
            else if (e.ValueKind == JsonValueKind.Array && e.GetArrayLength() == 2)
            {
                // transformers >= 4.45 pair format; spaces inside a part are encoded as U+0120 (chr(ord(' ') + 256))
                list[i++] = Fix(e[0].GetString()!) + " " + Fix(e[1].GetString()!);
            }
            else throw new InvalidDataException("Unknown tokenizer merges format.");
        }
        return list;

        static string Fix(string s) => s.Contains(' ') ? s.Replace(' ', 'Ġ') : s;
    }

    // ---- special tokens (gguf-py SpecialVocab) ------------------------------------------------
    static void AddSpecialVocab(IDictionary<string, object> md, string dir, HfConfig config, JsonElement tok, JsonElement? tcRoot, JsonElement[] addedTokens)
    {
        var types = new List<string> { "bos", "eos", "unk", "sep", "pad", "cls", "mask" };
        var ids = new Dictionary<string, int>();
        var add = new Dictionary<string, bool>();
        var overrides = new Dictionary<string, string?>(); // tokenizer_config entries rewritten by the post-processor logic
        bool hasTc = tcRoot is { ValueKind: JsonValueKind.Object } r0 && r0.EnumerateObject().Any();

        string? Tc(string key)
        {
            if (overrides.TryGetValue(key, out var o)) return o;
            if (tcRoot is not { ValueKind: JsonValueKind.Object } root || !root.TryGetProperty(key, out var e)) return null;
            return e.ValueKind switch
            {
                JsonValueKind.String => e.GetString(),
                JsonValueKind.Object when e.TryGetProperty("content", out var c) && c.ValueKind == JsonValueKind.String => c.GetString(),
                _ => null,
            };
        }
        static bool Has(string? s) => !string.IsNullOrEmpty(s);

        string? bos = Tc("bos_token"), cls = Tc("cls_token"), eos = Tc("eos_token"), sep = Tc("sep_token");
        if (!Has(bos) && Has(cls) && hasTc) overrides["bos_token"] = bos = cls;
        if (!Has(eos) && Has(sep) && hasTc) overrides["eos_token"] = eos = sep;

        if (tok.TryGetProperty("post_processor", out var pp) && pp.ValueKind == JsonValueKind.Object)
        {
            var processors = pp.TryGetProperty("processors", out var ps) && ps.ValueKind == JsonValueKind.Array ? ps.EnumerateArray().ToArray() : [pp];
            foreach (var proc in processors)
            {
                string ptype = proc.TryGetProperty("type", out var pt) ? pt.GetString() ?? "" : "";
                if (ptype == "ByteLevel")
                {
                    add.TryAdd("bos", false);
                    add.TryAdd("eos", false);
                }
                else if (ptype == "RobertaProcessing")
                {
                    add["bos"] = add["eos"] = add["sep"] = true;
                    if (!Has(cls) && hasTc && proc.TryGetProperty("cls", out var c) && c.ValueKind == JsonValueKind.Array)
                        overrides["cls_token"] = cls = c[0].GetString();
                    if (!Has(sep) && hasTc && proc.TryGetProperty("sep", out var s) && s.ValueKind == JsonValueKind.Array)
                        overrides["sep_token"] = sep = s[0].GetString();
                }
                else if (ptype == "TemplateProcessing")
                {
                    // Crude parsing of the single-sequence template to decide whether BOS/EOS are added.
                    // TODO: the pair template (separator tokens) is not interpreted.
                    if (!proc.TryGetProperty("single", out var single) || single.ValueKind != JsonValueKind.Array || single.GetArrayLength() <= 1) continue;
                    string? first = SpecialId(single[0]), last = SpecialId(single[single.GetArrayLength() - 1]);
                    if (Has(first))
                    {
                        if (!hasTc) bos = first;
                        else if (first != bos && first != cls)
                        {
                            if (!Has(bos)) overrides["bos_token"] = bos = first;
                            if (!Has(cls)) overrides["cls_token"] = cls = first;
                        }
                        add["bos"] = first == bos || first == cls;
                    }
                    if (Has(last))
                    {
                        if (!hasTc) eos = last;
                        else if (last != eos)
                        {
                            if (!types.Contains("eot")) { types.Add("eot"); overrides["eot_token"] = eos ?? ""; }
                            else if (!types.Contains("eom")) { types.Add("eom"); overrides["eom_token"] = eos ?? ""; }
                            overrides["eos_token"] = eos = last;
                        }
                        add["eos"] = last == eos;
                    }
                }
            }
        }

        static string? SpecialId(JsonElement e)
            => e.TryGetProperty("SpecialToken", out var st) && st.TryGetProperty("id", out var id) ? id.GetString() : null;

        void SetId(string typ, int id)
        {
            if (id < 0) throw new InvalidDataException($"invalid value for special token type {typ}: {id}");
            ids.TryAdd(typ, id);
        }

        if (hasTc)
        {
            string? template = ReadChatTemplate(dir, tcRoot!.Value);
            if (template != null) md[CanonicalKeys.ChatTemplate] = template;

            foreach (var typ in types.ToArray())
            {
                if (tcRoot!.Value.TryGetProperty($"add_{typ}_token", out var ab) && ab.ValueKind is JsonValueKind.True or JsonValueKind.False)
                    add.TryAdd(typ, ab.ValueKind == JsonValueKind.True);
                string? content = Tc($"{typ}_token");
                if (!Has(content)) continue;
                // like llama.cpp: only the added_tokens list is searched
                foreach (var a in addedTokens)
                    if (a.GetProperty("content").GetString() == content) { SetId(typ, a.GetProperty("id").GetInt32()); break; }
            }
        }

        // config.json fallback (root, then text_config)
        foreach (var typ in types)
        {
            string key = $"{typ}_token_id";
            int? id = null;
            if (config.Root.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.Number) id = v.GetInt32();
            else if (config.Root.TryGetProperty("text_config", out var tc2) && tc2.TryGetProperty(key, out var v2) && v2.ValueKind == JsonValueKind.Number) id = v2.GetInt32();
            if (id is { } i) SetId(typ, i);
        }

        // Only the kinds that have a canonical key (cls ids / add_cls have none in GGUF either).
        foreach (var (typ, id) in ids)
        {
            string? key = typ switch
            {
                "bos" => CanonicalKeys.TokenizerBosId, "eos" => CanonicalKeys.TokenizerEosId,
                "eot" => CanonicalKeys.TokenizerEotId, "eom" => CanonicalKeys.TokenizerEomId,
                "unk" => CanonicalKeys.TokenizerUnkId, "pad" => CanonicalKeys.TokenizerPadId,
                "sep" => "tokenizer.seperator_token_id", "mask" => "tokenizer.mask_token_id",
                _ => null,
            };
            if (key != null) md[key] = (uint)id;
        }
        foreach (var (typ, v) in add)
        {
            string? key = typ switch { "bos" => CanonicalKeys.TokenizerAddBos, "eos" => CanonicalKeys.TokenizerAddEos, "sep" => "tokenizer.add_sep_token", _ => null };
            if (key != null) md[key] = v;
        }
        // TODO: normalizer flags (NFC/lowercase/strip_accents), clean_up_tokenization_spaces and metaspace prepend scheme
        // are not part of the canonical vocabulary yet, so they are not emitted.
    }

    static string? ReadChatTemplate(string dir, JsonElement tcRoot)
    {
        string? template = null;
        if (tcRoot.TryGetProperty("chat_template", out var ct)) template = TemplateFrom(ct);
        string jinja = Path.Combine(dir, "chat_template.jinja"), json = Path.Combine(dir, "chat_template.json");
        if (File.Exists(jinja)) template = File.ReadAllText(jinja, Encoding.UTF8);
        else if (File.Exists(json))
        {
            using var d = JsonDocument.Parse(File.ReadAllBytes(json));
            if (d.RootElement.TryGetProperty("chat_template", out var c2)) template = TemplateFrom(c2);
        }
        return template;

        // A list of named templates collapses to the "default" (else first) one.
        static string? TemplateFrom(JsonElement e)
        {
            if (e.ValueKind == JsonValueKind.String) return e.GetString();
            if (e.ValueKind != JsonValueKind.Array) return null;
            string? first = null;
            foreach (var item in e.EnumerateArray())
            {
                if (!item.TryGetProperty("template", out var t) || t.ValueKind != JsonValueKind.String) continue;
                first ??= t.GetString();
                if (item.TryGetProperty("name", out var n) && n.GetString() == "default") return t.GetString();
            }
            return first;
        }
    }

    static JsonDocument? ReadOptional(string path)
        => File.Exists(path) ? JsonDocument.Parse(File.ReadAllBytes(path)) : null;
}
