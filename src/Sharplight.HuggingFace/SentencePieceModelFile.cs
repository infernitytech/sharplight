using System.Text;

namespace Sharplight.HuggingFace;

/// <summary>Minimal reader of a SentencePiece <c>tokenizer.model</c> (protobuf ModelProto): pieces, scores, types and the dummy-prefix flag.</summary>
internal sealed class SentencePieceModelFile
{
    // Protobuf wire types.
    private const int WireVarint = 0;
    private const int WireFixed64 = 1;
    private const int WireLengthDelimited = 2;
    private const int WireFixed32 = 5;

    // ModelProto fields.
    private const int ModelPiecesField = 1;
    private const int ModelNormalizerSpecField = 3;

    // SentencePiece fields.
    private const int PiecePieceField = 1;
    private const int PieceScoreField = 2;
    private const int PieceTypeField = 3;

    // NormalizerSpec fields.
    private const int NormalizerAddDummyPrefixField = 3;

    private const int NormalPieceType = 1;

    /// <summary>The vocabulary pieces in id order.</summary>
    public required string[] Pieces { get; init; }

    /// <summary>The unigram log-probability of each piece.</summary>
    public required float[] Scores { get; init; }

    /// <summary>SentencePiece piece types: 1 normal, 2 unknown, 3 control, 4 user defined, 5 unused, 6 byte.</summary>
    public required int[] Types { get; init; }

    /// <summary>Whether the normalizer prepends a space (▁) to the text; SentencePiece's default is true.</summary>
    public bool AddDummyPrefix { get; init; } = true;

    /// <summary>Reads the model file at <paramref name="path"/>.</summary>
    public static SentencePieceModelFile Read(string path)
    {
        var data = File.ReadAllBytes(path);
        var pieces = new List<string>();
        var scores = new List<float>();
        var types = new List<int>();
        bool dummyPrefix = true;
        int pos = 0;
        while (pos < data.Length)
        {
            ulong tag = Varint(data, ref pos);
            int field = (int)(tag >> 3);
            int wire = (int)(tag & 7);
            if (wire != WireLengthDelimited)
            {
                Skip(data, ref pos, wire);
                continue;
            }

            int length = (int)Varint(data, ref pos);
            var body = data.AsSpan(pos, length).ToArray();
            pos += length;
            if (field == ModelPiecesField)
            {
                ReadPiece(body, pieces, scores, types);
            }
            else if (field == ModelNormalizerSpecField)
            {
                dummyPrefix = ReadNormalizer(body, dummyPrefix);
            }
        }

        return new SentencePieceModelFile { Pieces = [.. pieces], Scores = [.. scores], Types = [.. types], AddDummyPrefix = dummyPrefix };
    }

    private static void ReadPiece(byte[] message, List<string> pieces, List<float> scores, List<int> types)
    {
        string piece = "";
        float score = 0;
        int type = NormalPieceType;
        int pos = 0;
        while (pos < message.Length)
        {
            ulong tag = Varint(message, ref pos);
            int field = (int)(tag >> 3);
            int wire = (int)(tag & 7);
            if (field == PiecePieceField && wire == WireLengthDelimited)
            {
                int length = (int)Varint(message, ref pos);
                piece = Encoding.UTF8.GetString(message, pos, length);
                pos += length;
            }
            else if (field == PieceScoreField && wire == WireFixed32)
            {
                score = BitConverter.ToSingle(message, pos);
                pos += 4;
            }
            else if (field == PieceTypeField && wire == WireVarint)
            {
                type = (int)Varint(message, ref pos);
            }
            else
            {
                Skip(message, ref pos, wire);
            }
        }

        pieces.Add(piece);
        scores.Add(score);
        types.Add(type);
    }

    private static bool ReadNormalizer(byte[] message, bool addDummyPrefix)
    {
        int pos = 0;
        while (pos < message.Length)
        {
            ulong tag = Varint(message, ref pos);
            int field = (int)(tag >> 3);
            int wire = (int)(tag & 7);
            if (field == NormalizerAddDummyPrefixField && wire == WireVarint)
            {
                addDummyPrefix = Varint(message, ref pos) != 0;
            }
            else
            {
                Skip(message, ref pos, wire);
            }
        }

        return addDummyPrefix;
    }

    private static ulong Varint(byte[] data, ref int pos)
    {
        ulong value = 0;
        int shift = 0;
        while (true)
        {
            byte b = data[pos++];
            value |= (ulong)(b & 0x7F) << shift;
            if ((b & 0x80) == 0)
            {
                return value;
            }

            shift += 7;
        }
    }

    private static void Skip(byte[] data, ref int pos, int wire)
    {
        switch (wire)
        {
            case WireVarint:
                Varint(data, ref pos);
                break;
            case WireFixed64:
                pos += 8;
                break;
            case WireLengthDelimited:
                int length = (int)Varint(data, ref pos);
                pos += length;
                break;
            case WireFixed32:
                pos += 4;
                break;
            default:
                throw new InvalidDataException($"Unsupported protobuf wire type {wire}.");
        }
    }
}
