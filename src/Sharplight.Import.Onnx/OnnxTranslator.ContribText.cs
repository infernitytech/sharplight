using Sharplight.Graph;

namespace Sharplight.Import.Onnx;

// com.microsoft text operators: WordConvEmbedding and MurmurHash3.
internal sealed partial class OnnxTranslator
{
    /// <summary>
    /// WordConvEmbedding: character ids [S, L] (0 = padding) → word embeddings [S, F]. Each word's characters are embedded (C [V, E]), a
    /// convolution of width K (W [F, 1, K, E], bias B [F]) with tanh runs over the first max(length, K) characters, where length counts the
    /// word's non-zero ids (0 when the first id is padding), and the result is max-pooled over the window positions; empty words embed to
    /// zeros (onnxruntime leaves them unwritten). Composed as one gather of the windows, one matmul and a masked max.
    /// </summary>
    private void WordConvEmbedding(OnnxNode n, TranslatedTensor?[] ins)
    {
        string name = Nm(n);
        var ids = Idx(ins[0]!);
        if (ids.Rank != 2 || Sym(ids))
        {
            throw new NotSupportedException("WordConvEmbedding expects a fixed [words, characters] sequence");
        }

        long words = ids.Shape[0].Fixed;
        long characters = ids.Shape[1].Fixed;
        var weight = X(ins, 1);
        long filters = weight.Shape[0].Fixed;
        long width = weight.Shape[2].Fixed;
        long embedding = weight.Shape[3].Fixed;
        long positions = characters - width + 1;
        if (positions <= 0)
        {
            throw new NotSupportedException($"WordConvEmbedding window {width} is wider than the {characters} characters per word");
        }

        var embedded = Emit("GatherAxis", [X(ins, 3), ids], Attributes.From(("axis", 0L)));                               // [S, L, E]
        var index = Rs(IndexConstant(name + ".windows", [.. Enumerable.Range(0, (int)(positions * width)).Select(i => (int)(i / width + i % width))]), positions, width);
        var windows = Rs(Emit("GatherAxis", [embedded, index], Attributes.From(("axis", 1L))), words, positions, width * embedding);
        var kernel = Synthetic("Transpose", [Synthetic("Reshape", [ins[1]!, TranslatedTensor.Const(OnnxConst.FromInts(OnnxType.Int64, [2], [filters, width * embedding]))])]);
        var conv = AsValue(Synthetic("MatMul", [TranslatedTensor.Dyn(windows), kernel]));                                 // [S, T, F]
        if (Opt(ins, 2) is { } bias)
        {
            conv = _b.Add(conv, AsValue(bias));
        }

        conv = _b.Unary("Tanh", conv);
        // Position t is inside the word when t + K <= length; position 0 whenever the word has a character.
        var present = _b.Binary("Greater", AsValue(ins[0]!), F32Const([0f], name + ".zero"));                               // [S, L]
        var counted = Emit("Reduce", [present], Attributes.From(("mode", "sum"), ("axes", L(1)), ("keepdims", true)));     // [S, 1]
        var lengths = _b.Mul(counted, _b.Slice(present, 1, 0, 1));   // a word that starts with padding is empty
        var limits = F32Const([.. Enumerable.Range(0, (int)positions).Select(t => t == 0 ? 1f : t + width)], name + ".limits");
        var inside = Rs(_b.Binary("LessEqual", limits, lengths), words, positions, 1);
        var masked = Emit("Where", [inside, conv, F32Const([float.NegativeInfinity], name + ".ninf")]);
        var pooled = Emit("Reduce", [masked], Attributes.From(("mode", "max"), ("axes", L(1)), ("keepdims", false)));     // [S, F]
        var hasCharacters = _b.Binary("Greater", lengths, F32Const([0f], name + ".none"));
        SetDyn(n, 0, Emit("Where", [hasCharacters, pooled, F32Const([0f], name + ".empty")], null, name));
    }

    /// <summary>
    /// MurmurHash3 (x86, 32 bit) of each element's bytes (4 for 32-bit integers, 8 for 64-bit ones, the UTF-8 bytes of a string) with the
    /// given seed; <c>positive</c> selects an unsigned result type.
    /// </summary>
    private void MurmurHash3(OnnxNode n, TranslatedTensor?[] ins)
    {
        var input = ins[0]!;
        bool isString = IsStringTensor(input);
        int elemType = isString ? OnnxType.String : ElemTypeOf(input);
        if (elemType is not (OnnxType.Int32 or OnnxType.UInt32 or OnnxType.Int64 or OnnxType.UInt64 or OnnxType.String))
        {
            throw new NotSupportedException($"MurmurHash3 of {OnnxType.Name(elemType)}");
        }

        var attributes = Attributes.From(("seed", n.Int("seed", 0)), ("element_bytes", elemType is OnnxType.Int64 or OnnxType.UInt64 ? 8L : 4L));
        var x = isString ? StrValue(input, Nm(n) + ".x") : Idx(input);
        // The signed and the unsigned (positive) hash share their bits; only the ONNX type differs.
        _env[n.Outputs[0]] = Typed(Emit("MurmurHash3", [x], attributes, Nm(n)), n.Int("positive", 1) != 0 ? OnnxType.UInt32 : OnnxType.Int32);
    }
}
