namespace Sharplight.Tokenizers.Bpe;

/// <summary>The GPT-2 reversible byte to printable-unicode-char mapping.</summary>
internal static class ByteEncoding
{
    public static readonly char[] ByteToChar = BuildByteToChar();

    /// <summary>Indexed by char; -1 for chars that are not part of the mapping.</summary>
    private static readonly short[] CharToByteTable = BuildCharToByte();

    private static char[] BuildByteToChar()
    {
        var map = new char[256];
        int n = 0;
        for (int b = 0; b < 256; b++)
        {
            bool direct = (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
            map[b] = direct ? (char)b : (char)(256 + n++);
        }
        return map;
    }

    private static short[] BuildCharToByte()
    {
        var t = new short[512];
        Array.Fill(t, (short)-1);
        for (int b = 0; b < 256; b++)
        {
            t[ByteToChar[b]] = (short)b;
        }

        return t;
    }

    public static bool TryGetByte(char c, out byte b)
    {
        short v = c < CharToByteTable.Length ? CharToByteTable[c] : (short)-1;
        b = (byte)v;
        return v >= 0;
    }
}
