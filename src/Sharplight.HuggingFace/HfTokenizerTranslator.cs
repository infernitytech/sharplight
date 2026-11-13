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
        // SentencePiece-style (Llama 2, Mistral, Gemma, Phi-3): BPE with byte fallback, scores come from tokenizer.model.
        if (modelType == "BPE" && model.TryGetProperty("byte_fallback", out var bf) && bf.ValueKind == JsonValueKind.True
            && File.Exists(Path.Combine(dir, "tokenizer.model")))
        {
            AddSentencePiece(md, dir, config, tok);
            return;
        }
        if (modelType is "WordPiece" or "Unigram")
        {
            AddVocabularyModel(md, dir, config, tok, model, modelType);
            return;
        }
        if (modelType != "BPE") throw new NotSupportedException($"tokenizer.json model type '{modelType}' is not supported (BPE, WordPiece, Unigram).");

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

    /// <summary>
    /// SentencePiece vocabulary (llama.cpp <c>_create_vocab_sentencepiece</c>): pieces, scores and types from
    /// <c>tokenizer.model</c>, extra added tokens appended into unused slots, padded to the model's vocab_size.
    /// </summary>
    static void AddSentencePiece(IDictionary<string, object> md, string dir, HfConfig config, JsonElement tok)
    {
        var sp = SentencePieceModelFile.Read(Path.Combine(dir, "tokenizer.model"));
        int vocabSize = config.GetInt("vocab_size") ?? sp.Pieces.Length;
        var tokens = new string[vocabSize]; var scores = new float[vocabSize]; var types = new int[vocabSize];
        for (int i = 0; i < vocabSize; i++)
        {
            if (i < sp.Pieces.Length) { tokens[i] = sp.Pieces[i]; scores[i] = sp.Scores[i]; types[i] = sp.Types[i]; }
            else { tokens[i] = $"[PAD{i}]"; scores[i] = -1000f; types[i] = TypeUnused; }
        }
        var addedTokens = tok.TryGetProperty("added_tokens", out var at) && at.ValueKind == JsonValueKind.Array ? at.EnumerateArray().ToArray() : [];
        var existing = new HashSet<string>(tokens, StringComparer.Ordinal);
        foreach (var a in addedTokens)
        {
            int id = a.GetProperty("id").GetInt32();
            string content = a.GetProperty("content").GetString()!;
            if (id >= vocabSize || existing.Contains(content) || types[id] != TypeUnused) continue;   // only fills unused slots
            tokens[id] = content; scores[id] = -1000f;
            types[id] = a.TryGetProperty("special", out var spc) && spc.ValueKind == JsonValueKind.True ? TypeControl : TypeUserDefined;
        }
        md[CanonicalKeys.TokenizerModel] = "spm";
        md[CanonicalKeys.TokenizerPre] = "default";
        md[CanonicalKeys.TokenizerTokens] = tokens;
        md[CanonicalKeys.TokenizerScores] = scores;
        md[CanonicalKeys.TokenizerTokenTypes] = types;
        md[CanonicalKeys.TokenizerAddSpacePrefix] = sp.AddDummyPrefix;
        JsonElement? tcRoot = null;
        using var tcDoc = ReadOptional(Path.Combine(dir, "tokenizer_config.json"));
        if (tcDoc != null) tcRoot = tcDoc.RootElement.Clone();
        AddSpecialVocab(md, dir, config, tok, tcRoot, addedTokens);
    }

    /// <summary>
    /// WordPiece (BERT family) and Unigram (SentencePiece unigram, XLM-RoBERTa family) vocabularies: tokens in id order (Hugging
    /// Face form: WordPiece continuations keep their <c>##</c>, unigram pieces their ▁), unigram scores, normalizer flags and the
    /// pair layout of cross-encoders.
    /// </summary>
    static void AddVocabularyModel(IDictionary<string, object> md, string dir, HfConfig config, JsonElement tok, JsonElement model, string modelType)
    {
        var addedTokens = tok.TryGetProperty("added_tokens", out var at) && at.ValueKind == JsonValueKind.Array ? at.EnumerateArray().ToArray() : [];
        var pieces = new List<string>();
        var scores = new List<float>();
        if (modelType == "WordPiece")
        {
            var vocab = model.GetProperty("vocab").EnumerateObject().Select(p => (p.Name, Id: p.Value.GetInt32())).ToList();
            int n = vocab.Count == 0 ? 0 : vocab.Max(v => v.Id) + 1;
            pieces.AddRange(Enumerable.Repeat("", n));
            foreach (var (name, id) in vocab) pieces[id] = name;
        }
        else
        {
            foreach (var e in model.GetProperty("vocab").EnumerateArray()) { pieces.Add(e[0].GetString()!); scores.Add(e[1].GetSingle()); }
            if (model.TryGetProperty("unk_id", out var uid) && uid.ValueKind == JsonValueKind.Number) md[CanonicalKeys.TokenizerUnkId] = (uint)uid.GetInt32();
        }
        int vocabSize = Math.Max(config.GetInt("vocab_size") ?? pieces.Count, pieces.Count);
        var tokens = new string[vocabSize];
        var types = new int[vocabSize];
        var sc = new float[vocabSize];
        for (int i = 0; i < vocabSize; i++)
        {
            if (i < pieces.Count && pieces[i].Length > 0) { tokens[i] = pieces[i]; types[i] = TypeNormal; sc[i] = i < scores.Count ? scores[i] : 0; }
            else { tokens[i] = $"[PAD{i}]"; types[i] = TypeUnused; sc[i] = -1000f; }
        }
        foreach (var a in addedTokens)
        {
            int id = a.GetProperty("id").GetInt32();
            if (id >= vocabSize) continue;
            tokens[id] = a.GetProperty("content").GetString()!;
            bool special = a.TryGetProperty("special", out var sp) && sp.ValueKind == JsonValueKind.True;
            types[id] = special ? TypeControl : TypeUserDefined;
        }
        md[CanonicalKeys.TokenizerModel] = modelType == "WordPiece" ? "wordpiece" : "unigram";
        md[CanonicalKeys.TokenizerPre] = "default";
        md[CanonicalKeys.TokenizerTokens] = tokens;
        md[CanonicalKeys.TokenizerTokenTypes] = types;
        if (modelType == "Unigram") md[CanonicalKeys.TokenizerScores] = sc;

        // Normalizer flags (BertNormalizer: lowercase / strip_accents; a Lowercase step in a sequence).
        bool lower = false; bool? strip = null;
        if (tok.TryGetProperty("normalizer", out var norm) && norm.ValueKind == JsonValueKind.Object) Visit(norm);
        void Visit(JsonElement n)
        {
            string type = n.TryGetProperty("type", out var t) ? t.GetString() ?? "" : "";
            if (type == "Sequence" && n.TryGetProperty("normalizers", out var arr)) foreach (var e in arr.EnumerateArray()) Visit(e);
            else if (type == "BertNormalizer")
            {
                lower |= n.TryGetProperty("lowercase", out var lc) && lc.ValueKind == JsonValueKind.True;
                if (n.TryGetProperty("strip_accents", out var sa) && sa.ValueKind is JsonValueKind.True or JsonValueKind.False) strip = sa.ValueKind == JsonValueKind.True;
            }
            else if (type == "Lowercase") lower = true;
            else if (type == "StripAccents") strip = true;
        }
        if (modelType == "WordPiece")
        {
            md[CanonicalKeys.TokenizerLowercase] = lower;
            if (strip is bool st) md[CanonicalKeys.TokenizerStripAccents] = st;
        }
        else
        {
            bool prefix = true;
            if (tok.TryGetProperty("pre_tokenizer", out var pre) && pre.ValueKind == JsonValueKind.Object && FindMetaspace(pre) is { } ms)
                prefix = ms.TryGetProperty("prepend_scheme", out var ps) ? ps.GetString() != "never"
                       : !ms.TryGetProperty("add_prefix_space", out var aps) || aps.ValueKind != JsonValueKind.False;
            md[CanonicalKeys.TokenizerAddSpacePrefix] = prefix;
        }

        // Pair layout: two separators between the texts (RoBERTa) or one with token type 1 for the second text (BERT).
        if (tok.TryGetProperty("post_processor", out var pp) && pp.ValueKind == JsonValueKind.Object)
        {
            string ptype = pp.TryGetProperty("type", out var ptt) ? ptt.GetString() ?? "" : "";
            if (ptype == "RobertaProcessing") md[CanonicalKeys.TokenizerPairStyle] = "roberta";
            else if (ptype == "TemplateProcessing" && pp.TryGetProperty("pair", out var pair) && pair.ValueKind == JsonValueKind.Array)
            {
                var items = pair.EnumerateArray().ToList();
                int a = items.FindIndex(e => e.TryGetProperty("Sequence", out var sq) && sq.GetProperty("id").GetString() == "A");
                int b = items.FindIndex(e => e.TryGetProperty("Sequence", out var sq) && sq.GetProperty("id").GetString() == "B");
                if (a >= 0 && b > a) md[CanonicalKeys.TokenizerPairStyle] = b - a - 1 >= 2 ? "roberta" : "bert";
            }
        }
        JsonElement? tcRoot = null;
        using var tcDoc = ReadOptional(Path.Combine(dir, "tokenizer_config.json"));
        if (tcDoc != null) tcRoot = tcDoc.RootElement.Clone();
        if (tcRoot is { ValueKind: JsonValueKind.Object } r && r.TryGetProperty("model_max_length", out var mml) && mml.ValueKind == JsonValueKind.Number
            && mml.TryGetInt64(out long ml) && ml is > 0 and < 1_000_000)
            md[CanonicalKeys.TokenizerMaxLength] = (int)ml;
        AddSpecialVocab(md, dir, config, tok, tcRoot, addedTokens);

        static JsonElement? FindMetaspace(JsonElement p)
        {
            string type = p.TryGetProperty("type", out var t) ? t.GetString() ?? "" : "";
            if (type == "Metaspace") return p;
            if (type == "Sequence" && p.TryGetProperty("pretokenizers", out var arr))
                foreach (var e in arr.EnumerateArray()) if (FindMetaspace(e) is { } m) return m;
            return null;
        }
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

        // Only the kinds that have a canonical key.
        foreach (var (typ, id) in ids)
        {
            string? key = typ switch
            {
                "bos" => CanonicalKeys.TokenizerBosId, "eos" => CanonicalKeys.TokenizerEosId,
                "eot" => CanonicalKeys.TokenizerEotId, "eom" => CanonicalKeys.TokenizerEomId,
                "unk" => CanonicalKeys.TokenizerUnkId, "pad" => CanonicalKeys.TokenizerPadId,
                "sep" => CanonicalKeys.TokenizerSeparatorId, "mask" => CanonicalKeys.TokenizerMaskId, "cls" => CanonicalKeys.TokenizerClassifierId,
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
