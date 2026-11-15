namespace Sharplight.Import.PyTorch;

/// <summary>Pickle opcode bytes (protocols 0 to 5) that the restricted unpickler knows by name.</summary>
internal static class PickleOp
{
    public const byte Mark = (byte)'(';
    public const byte Stop = (byte)'.';
    public const byte Pop = (byte)'0';
    public const byte PopMark = (byte)'1';
    public const byte Dup = (byte)'2';
    public const byte Float = (byte)'F';
    public const byte Int = (byte)'I';
    public const byte BinInt = (byte)'J';
    public const byte BinInt1 = (byte)'K';
    public const byte Long = (byte)'L';
    public const byte BinInt2 = (byte)'M';
    public const byte None = (byte)'N';
    public const byte PersId = (byte)'P';
    public const byte BinPersId = (byte)'Q';
    public const byte Reduce = (byte)'R';
    public const byte String = (byte)'S';
    public const byte BinString = (byte)'T';
    public const byte ShortBinString = (byte)'U';
    public const byte Unicode = (byte)'V';
    public const byte BinUnicode = (byte)'X';
    public const byte Append = (byte)'a';
    public const byte Build = (byte)'b';
    public const byte Global = (byte)'c';
    public const byte Dict = (byte)'d';
    public const byte EmptyDict = (byte)'}';
    public const byte Appends = (byte)'e';
    public const byte Get = (byte)'g';
    public const byte BinGet = (byte)'h';
    public const byte Inst = (byte)'i';
    public const byte LongBinGet = (byte)'j';
    public const byte List = (byte)'l';
    public const byte EmptyList = (byte)']';
    public const byte Obj = (byte)'o';
    public const byte Put = (byte)'p';
    public const byte BinPut = (byte)'q';
    public const byte LongBinPut = (byte)'r';
    public const byte SetItem = (byte)'s';
    public const byte Tuple = (byte)'t';
    public const byte EmptyTuple = (byte)')';
    public const byte SetItems = (byte)'u';
    public const byte BinFloat = (byte)'G';
    public const byte Proto = 0x80;
    public const byte NewObj = 0x81;
    public const byte Ext1 = 0x82;
    public const byte Ext2 = 0x83;
    public const byte Ext4 = 0x84;
    public const byte Tuple1 = 0x85;
    public const byte Tuple2 = 0x86;
    public const byte Tuple3 = 0x87;
    public const byte NewTrue = 0x88;
    public const byte NewFalse = 0x89;
    public const byte Long1 = 0x8a;
    public const byte Long4 = 0x8b;
    public const byte BinBytes = (byte)'B';
    public const byte ShortBinBytes = (byte)'C';
    public const byte ShortBinUnicode = 0x8c;
    public const byte BinUnicode8 = 0x8d;
    public const byte BinBytes8 = 0x8e;
    public const byte EmptySet = 0x8f;
    public const byte AddItems = 0x90;
    public const byte FrozenSet = 0x91;
    public const byte NewObjEx = 0x92;
    public const byte StackGlobal = 0x93;
    public const byte Memoize = 0x94;
    public const byte Frame = 0x95;
    public const byte ByteArray8 = 0x96;
    public const byte NextBuffer = 0x97;
    public const byte ReadOnlyBuffer = 0x98;

    /// <summary>The Python name of a refused opcode, for error messages.</summary>
    public static string Describe(byte op) => op switch
    {
        Inst => "INST",
        Obj => "OBJ",
        NewObj => "NEWOBJ",
        NewObjEx => "NEWOBJ_EX",
        Ext1 => "EXT1",
        Ext2 => "EXT2",
        Ext4 => "EXT4",
        PersId => "PERSID",
        ByteArray8 => "BYTEARRAY8",
        NextBuffer => "NEXT_BUFFER",
        ReadOnlyBuffer => "READONLY_BUFFER",
        EmptySet => "EMPTY_SET",
        AddItems => "ADDITEMS",
        FrozenSet => "FROZENSET",
        Float => "FLOAT",
        String => "STRING",
        Unicode => "UNICODE",
        Dup => "DUP",
        Get => "GET",
        Put => "PUT",
        _ => $"0x{op:x2}",
    };
}
