#!/bin/bash
# The affected suites: Core / Hosting on the CPU; Cuda / Serving on CUDA; Vulkan.Tests attention classes on CUDA and Vulkan.
D=$(dirname "$0")
TAG=cpu SUITES="Sharplight.Core.Tests Sharplight.Hosting.Tests" flock ~/.sl-bench.lock $D/tests.sh
TAG=cuda flock ~/.sl-bench.lock $D/tests.sh
V=tests/Sharplight.Vulkan.Tests
F="$V/PagedAttentionTests.cs $V/FullAttentionTests.cs $V/SparseAttentionTests.cs $V/StatefulTests.cs $V/CompressedKvTests.cs $V/FusionTests.cs"
TAG=vk-cuda SUITES=Sharplight.Vulkan.Tests FILES="$F" flock ~/.sl-bench.lock $D/tests.sh
TAG=vk-vulkan SHARPLIGHT_TEST_PROVIDER=vulkan SUITES=Sharplight.Vulkan.Tests FILES="$F" flock ~/.sl-bench.lock $D/tests.sh
