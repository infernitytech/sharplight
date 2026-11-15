#pragma once
#include <cstdint>
// Stand-ins for the CUTLASS types the preprocessor reads (values of mixed_gemm_B_layout.h of EETQ, Sm75 / Sm80 branch).
namespace cutlass {
struct uint4b_t {};
namespace layout { struct RowMajor {}; struct ColumnMajor {}; template<int R, int C> struct ColumnMajorTileInterleave {}; }
namespace arch { struct Sm70 {}; struct Sm75 {}; struct Sm80 {}; struct OpMultiplyAdd {}; struct OpMultiplyAddDequantizeInterleavedBToA {}; }
namespace gemm { namespace kernel {
template<typename TypeB, typename Arch> struct LayoutDetailsB {};
template<typename TypeB> struct LayoutDetailsB<TypeB, arch::Sm70> { using Layout = layout::RowMajor; using Operator = arch::OpMultiplyAdd; };
template<> struct LayoutDetailsB<uint8_t, arch::Sm75> { using Layout = layout::ColumnMajorTileInterleave<64, 2>; using Operator = arch::OpMultiplyAddDequantizeInterleavedBToA; };
template<> struct LayoutDetailsB<uint8_t, arch::Sm80> { using Layout = layout::ColumnMajorTileInterleave<64, 2>; using Operator = arch::OpMultiplyAddDequantizeInterleavedBToA; };
template<> struct LayoutDetailsB<uint4b_t, arch::Sm75> { using Layout = layout::ColumnMajorTileInterleave<64, 4>; using Operator = arch::OpMultiplyAddDequantizeInterleavedBToA; };
template<> struct LayoutDetailsB<uint4b_t, arch::Sm80> { using Layout = layout::ColumnMajorTileInterleave<64, 4>; using Operator = arch::OpMultiplyAddDequantizeInterleavedBToA; };
}}}
