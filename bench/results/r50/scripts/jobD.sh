#!/bin/bash
# GPU job D: final branch prepare probe (sized memo table), then the Core / Serving / Cuda / Hosting suites on GPU 0.
. <path>/env.sh
export HOME=<home>/test-home NUGET_PACKAGES=<path>/packages DOTNET_CLI_HOME=<home> SHARPLIGHT_VK_DEVICES=0 CUDA_VISIBLE_DEVICES=0
L=<path>/Llama-3.1-8B-Instruct; O=<path>/br2; mkdir -p $O
uptime > $O/probe_llama.load
timeout 900 <path>/BatchBench $L --provider cuda --slots 256 --ctx 64 --depth 16 --batch 2048 --sizes 256 --prepare-probe --prepare-rounds 4 > $O/probe_llama.txt 2>&1
DOTNET_ROLL_FORWARD=Major timeout 900 <path>/dotnet-trace collect --profile dotnet-sampled-thread-time --format Speedscope -o $O/probe_llama.nettrace -- <path>/BatchBench $L --provider cuda --slots 256 --ctx 64 --depth 16 --batch 2048 --sizes 256 --prepare-probe --prepare-rounds 8 > $O/probe_llama_trace.txt 2>&1
SHARPLIGHT_TEST_PROVIDER=cuda SUITE_JOBS=2 <path>/run-suites.sh <path>/perf-r50 <path>/tests Core.Tests Serving.Tests Cuda.Tests Hosting.Tests > <path>/tests.log 2>&1
echo jobD-done
