using System.Runtime.InteropServices;

namespace Sharplight.Vulkan;

/// <summary>
/// SPIR-V rewrites for layered Vulkan implementations that translate kernels to another shading language (MoltenVK translates
/// them to MSL with SPIRV-Cross). Applied only on devices that report <c>VK_KHR_portability_subset</c>; every other device gets
/// the kernels unchanged. Both rewrites keep the module's meaning and only avoid constructs SPIRV-Cross translates wrongly for
/// buffer-device-address pointers:
/// <list type="bullet">
/// <item><c>OpCopyLogical</c> of a struct with pointer members (Slang copies every push-constant parameter block from its laid-out
/// type to a logical twin) is translated member by member with a dereference of each pointer member
/// (<c>*_41.x = entryPointParams.p.x</c>). It becomes <c>OpCompositeExtract</c> of every member and one
/// <c>OpCompositeConstruct</c>, which SPIRV-Cross emits as plain value copies.</item>
/// <item>An <c>OpPtrAccessChain</c> result used as a pointer value (cast, passed, returned, stored or selected) is emitted as
/// the element value instead of its address (<c>reinterpret_cast&lt;device float4*&gt;(_70.o[i])</c>). Such access chains
/// become address arithmetic: <c>ConvertUToPtr(ConvertPtrToU(base) + index * ArrayStride)</c>.</item>
/// </list>
/// </summary>
internal static class SpirvPortability
{
    private const uint MagicNumber = 0x07230203;
    private const int HeaderWords = 5;
    private const int BoundWord = 3;
    private const uint StorageClassPhysicalStorageBuffer = 5349;
    private const uint DecorationArrayStride = 6;

    private const ushort OpTypeInt = 21;
    private const ushort OpTypeStruct = 30;
    private const ushort OpTypePointer = 32;
    private const ushort OpConstant = 43;
    private const ushort OpFunction = 54;
    private const ushort OpFunctionCall = 57;
    private const ushort OpStore = 62;
    private const ushort OpSelect = 169;
    private const ushort OpPhi = 245;
    private const ushort OpReturnValue = 254;
    private const ushort OpPtrAccessChain = 67;
    private const ushort OpDecorate = 71;
    private const ushort OpCompositeConstruct = 80;
    private const ushort OpCompositeExtract = 81;
    private const ushort OpUConvert = 113;
    private const ushort OpSConvert = 114;
    private const ushort OpConvertPtrToU = 117;
    private const ushort OpConvertUToPtr = 120;
    private const ushort OpBitcast = 124;
    private const ushort OpIAdd = 128;
    private const ushort OpIMul = 132;
    private const ushort OpCopyLogical = 400;

    /// <summary>Returns <paramref name="code"/> with the rewrites above, or <paramref name="code"/> itself when none apply.</summary>
    public static byte[] Rewrite(byte[] code)
    {
        if (code.Length < HeaderWords * 4 || code.Length % 4 != 0)
        {
            return code;
        }

        var words = MemoryMarshal.Cast<byte, uint>(code.AsSpan()).ToArray();
        if (words[0] != MagicNumber)
        {
            return code;
        }

        var module = new Module(words);
        var rewritten = module.Rewrite();
        return rewritten is null ? code : MemoryMarshal.AsBytes(rewritten.AsSpan()).ToArray();
    }

    /// <summary>One parsed module: instruction offsets, type facts and the id bound for new instructions.</summary>
    private sealed class Module
    {
        private readonly uint[] _words;
        private readonly List<int> _instructions = [];
        private readonly Dictionary<uint, uint[]> _structs = [];
        private readonly Dictionary<uint, (uint Width, bool Signed)> _ints = [];
        private readonly HashSet<uint> _bufferPointers = [];
        private readonly Dictionary<uint, uint> _arrayStrides = [];
        private readonly Dictionary<uint, uint> _typeOf = [];
        private readonly HashSet<uint> _pointerValues = [];
        private readonly Dictionary<ulong, uint> _ulongConstants = [];
        private readonly List<uint> _newGlobals = [];
        private uint _bound;
        private uint _ulong;

        public Module(uint[] words)
        {
            _words = words;
            _bound = words[BoundWord];
            Parse();
        }

        /// <summary>The rewritten words, or null when the module has nothing to rewrite.</summary>
        public uint[]? Rewrite()
        {
            var body = new List<uint>(_words.Length + 64);
            int functionsStart = -1;
            bool changed = false;
            foreach (int at in _instructions)
            {
                ushort op = (ushort)_words[at];
                int count = (int)(_words[at] >> 16);
                if (op == OpFunction && functionsStart < 0)
                {
                    functionsStart = body.Count;
                }

                if (op == OpCopyLogical && TrySplitCopy(at, body))
                {
                    changed = true;
                    continue;
                }

                if (op == OpPtrAccessChain && TryAddressArithmetic(at, count, body))
                {
                    changed = true;
                    continue;
                }

                body.AddRange(_words.AsSpan(at, count));
            }

            if (!changed)
            {
                return null;
            }

            // New types and constants go after every existing global declaration, before the first function.
            body.InsertRange(functionsStart < 0 ? body.Count : functionsStart, _newGlobals);
            var header = _words.AsSpan(0, HeaderWords).ToArray();
            header[BoundWord] = _bound;
            return [.. header, .. body];
        }

        private void Parse()
        {
            int at = HeaderWords;
            while (at < _words.Length)
            {
                int count = (int)(_words[at] >> 16);
                if (count == 0 || at + count > _words.Length)
                {
                    break;
                }

                _instructions.Add(at);
                ushort op = (ushort)_words[at];
                switch (op)
                {
                    case OpTypeInt:
                        _ints[_words[at + 1]] = (_words[at + 2], _words[at + 3] != 0);
                        if (_words[at + 2] == 64 && _words[at + 3] == 0)
                        {
                            _ulong = _words[at + 1];
                        }

                        break;
                    case OpTypeStruct:
                        _structs[_words[at + 1]] = _words.AsSpan(at + 2, count - 2).ToArray();
                        break;
                    case OpTypePointer when _words[at + 2] == StorageClassPhysicalStorageBuffer:
                        _bufferPointers.Add(_words[at + 1]);
                        break;
                    case OpDecorate when count == 4 && _words[at + 2] == DecorationArrayStride:
                        _arrayStrides[_words[at + 1]] = _words[at + 3];
                        break;
                    case OpBitcast or OpConvertPtrToU:
                        _pointerValues.Add(_words[at + 3]);
                        break;
                    case OpReturnValue:
                        _pointerValues.Add(_words[at + 1]);
                        break;
                    case OpStore when count >= 3:
                        _pointerValues.Add(_words[at + 2]);
                        break;
                    case OpFunctionCall:
                        _pointerValues.UnionWith(_words.AsSpan(at + 4, count - 4).ToArray());
                        break;
                    case OpSelect when count == 6:
                        _pointerValues.Add(_words[at + 4]);
                        _pointerValues.Add(_words[at + 5]);
                        break;
                    case OpPhi:
                        for (int i = at + 3; i < at + count; i += 2)
                        {
                            _pointerValues.Add(_words[i]);
                        }

                        break;
                }

                at += count;
            }

            // Result types of value instructions: word 1 is a type id and word 2 the result id.
            var types = new HashSet<uint>(_ints.Keys.Concat(_structs.Keys).Concat(_bufferPointers));
            foreach (int i in _instructions)
            {
                ushort op = (ushort)_words[i];
                bool hasResultType = op > 39 && op is not (>= 71 and <= 78) and not (>= 330 and <= 332) and not (5632 or 5633);
                if (hasResultType && (_words[i] >> 16) >= 3 && types.Contains(_words[i + 1]))
                {
                    _typeOf[_words[i + 2]] = _words[i + 1];
                }
            }
        }

        /// <summary>OpCopyLogical between structs with identical member types: extract every member, construct the copy.</summary>
        private bool TrySplitCopy(int at, List<uint> body)
        {
            uint resultType = _words[at + 1];
            uint result = _words[at + 2];
            uint operand = _words[at + 3];
            if (!_typeOf.TryGetValue(operand, out uint operandType) || !_structs.TryGetValue(resultType, out var members)
                || !_structs.TryGetValue(operandType, out var operandMembers) || !members.AsSpan().SequenceEqual(operandMembers))
            {
                return false;
            }

            var parts = new uint[members.Length];
            for (int i = 0; i < members.Length; i++)
            {
                parts[i] = _bound++;
                body.AddRange([Word(OpCompositeExtract, 5), members[i], parts[i], operand, (uint)i]);
            }

            body.AddRange([Word(OpCompositeConstruct, 3 + members.Length), resultType, result, .. parts]);
            return true;
        }

        /// <summary>A single-index OpPtrAccessChain on a buffer pointer whose result is used as a value, as address arithmetic.</summary>
        private bool TryAddressArithmetic(int at, int count, List<uint> body)
        {
            uint resultType = _words[at + 1];
            uint result = _words[at + 2];
            if (count != 5 || !_pointerValues.Contains(result) || !_bufferPointers.Contains(resultType)
                || !_arrayStrides.TryGetValue(resultType, out uint stride)
                || !_typeOf.TryGetValue(_words[at + 4], out uint indexType) || !_ints.TryGetValue(indexType, out var index))
            {
                return false;
            }

            uint ulongType = UlongType();
            uint baseAddress = _bound++;
            body.AddRange([Word(OpConvertPtrToU, 4), ulongType, baseAddress, _words[at + 3]]);
            uint element = _words[at + 4];
            if (index.Width != 64)
            {
                uint widened = _bound++;
                body.AddRange([Word(index.Signed ? OpSConvert : OpUConvert, 4), ulongType, widened, element]);
                element = widened;
            }

            uint offset = _bound++;
            body.AddRange([Word(OpIMul, 5), ulongType, offset, element, UlongConstant(stride)]);
            uint address = _bound++;
            body.AddRange([Word(OpIAdd, 5), ulongType, address, baseAddress, offset]);
            body.AddRange([Word(OpConvertUToPtr, 4), resultType, result, address]);
            return true;
        }

        private uint UlongType()
        {
            if (_ulong == 0)
            {
                _ulong = _bound++;
                _newGlobals.AddRange([Word(OpTypeInt, 4), _ulong, 64, 0]);
            }

            return _ulong;
        }

        private uint UlongConstant(ulong value)
        {
            if (!_ulongConstants.TryGetValue(value, out uint id))
            {
                id = _bound++;
                _newGlobals.AddRange([Word(OpConstant, 5), UlongType(), id, (uint)value, (uint)(value >> 32)]);
                _ulongConstants[value] = id;
            }

            return id;
        }

        private static uint Word(ushort op, int count) => ((uint)count << 16) | op;
    }
}
