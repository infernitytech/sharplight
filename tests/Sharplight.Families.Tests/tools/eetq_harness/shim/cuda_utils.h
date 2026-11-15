#pragma once
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
typedef _Float16 half;
#define FT_CHECK_WITH_INFO(cond, info) do { if (!(cond)) { std::fprintf(stderr, "FT_CHECK failed: %s\n", std::string(info).c_str()); std::abort(); } } while (0)
#define FT_CHECK(cond) FT_CHECK_WITH_INFO(cond, #cond)
template <typename... A> std::string fmtstr(const char* f, A... a) { char b[512]; std::snprintf(b, sizeof b, f, a...); return b; }
namespace fastertransformer { inline int getSMVersion() { return 80; } }
