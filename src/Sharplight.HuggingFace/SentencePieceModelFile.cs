using System.Text;

namespace Sharplight.HuggingFace;

/// <summary>Minimal reader of a SentencePiece <c>tokenizer.model</c> (protobuf ModelProto): pieces, scores, types and the dummy-prefix flag.</summary>
public sealed class SentencePieceModelFile
{
    public required string[] Pieces { get; init; }
    public required float[] Scores { get; init; }
    /// <summary>SentencePiece piece types: 1 normal, 2 unknown, 3 control, 4 user defined, 5 unused, 6 byte.</summary>
    public required int[] Types { get; init; }
    public bool AddDummyPrefix { get; init; } = true;

    public static SentencePieceModelFile Read(string path)
    {
        var data = File.ReadAllBytes(path);
        var pieces = new List<string>(); var scores = new List<float>(); var types = new List<int>();
        bool dummyPrefix = true;
        int pos = 0;
        while (pos < data.Length)
        {
            ulong tag = Varint(data, ref pos);
            int field = (int)(tag >> 3), wire = (int)(tag & 7);
            if (wire != 2) { Skip(data, ref pos, wire); continue; }
            int len = (int)Varint(data, ref pos);
            var body = data.AsSpan(pos, len).ToArray();
            pos += len;
            if (field == 1)
            {
                ReadPiece(body, pieces, scores, types);
            }
            else if (field == 3)
            {
                dummyPrefix = ReadNormalizer(body, dummyPrefix);
            }
        }
        return new SentencePieceModelFile { Pieces = [.. pieces], Scores = [.. scores], Types = [.. types], AddDummyPrefix = dummyPrefix };
    }

    private static void ReadPiece(byte[] arr, List<string> pieces, List<float> scores, List<int> types)
    {
        string piece = ""; float score = 0; int type = 1;
        int pos = 0;
        while (pos < arr.Length)
        {
            ulong tag = Varint(arr, ref pos);
            int field = (int)(tag >> 3), wire = (int)(tag & 7);
            if (field == 1 && wire == 2) { int l = (int)Varint(arr, ref pos); piece = Encoding.UTF8.GetString(arr, pos, l); pos += l; }
            else if (field == 2 && wire == 5) { score = BitConverter.ToSingle(arr, pos); pos += 4; }
            else if (field == 3 && wire == 0)
            {
                type = (int)Varint(arr, ref pos);
            }
            else
            {
                Skip(arr, ref pos, wire);
            }
        }
        pieces.Add(piece); scores.Add(score); types.Add(type);
    }

    private static bool ReadNormalizer(byte[] arr, bool dummy)
    {
        int pos = 0;
        while (pos < arr.Length)
        {
            ulong tag = Varint(arr, ref pos);
            int field = (int)(tag >> 3), wire = (int)(tag & 7);
            if (field == 3 && wire == 0)
            {
                dummy = Varint(arr, ref pos) != 0;
            }
            else
            {
                Skip(arr, ref pos, wire);
            }
        }
        return dummy;
    }

    private static ulong Varint(byte[] d, ref int pos)
    {
        ulong v = 0; int shift = 0;
        while (true)
        {
            byte x = d[pos++];
            v |= (ulong)(x & 0x7F) << shift;
            if ((x & 0x80) == 0)
            {
                return v;
            }

            shift += 7;
        }
    }

    private static void Skip(byte[] d, ref int pos, int wire)
    {
        switch (wire)
        {
            case 0: Varint(d, ref pos); break;
            case 1: pos += 8; break;
            case 2: { int l = (int)Varint(d, ref pos); pos += l; break; }
            case 5: pos += 4; break;
            default: throw new InvalidDataException($"Unsupported protobuf wire type {wire}.");
        }
    }
}
