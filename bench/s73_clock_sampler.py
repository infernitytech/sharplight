"""Samples the SM / memory clock and the power draw of one GPU through NVML at a high rate while a command runs.

Usage: python bench/s73_clock_sampler.py --gpu 0 --out samples.csv -- <command ...>

Starts the command, samples every --interval seconds (default 2 ms; NVML refreshes the power reading about every 10-20 ms,
the clocks on each query) until it exits, writes time,sm_mhz,mem_mhz,power_w,temp_c,reasons,instant_w rows to --out (reasons: NVML's clock
event reason bits, e.g. 0x4 software power cap, 0x20 software thermal slowdown, 0x40 hardware thermal, 0x80 hardware power brake) and
prints the median SM clock and power, and the share of samples with each reason, over the samples whose power is above --busy watts (the kernel's sustained window, not its start-up). The GPU index
is NVML's (physical) index; CUDA_VISIBLE_DEVICES of the command is left as it is.
"""

import argparse
import statistics
import subprocess
import sys
import time

import pynvml


def instant_power(handle) -> float:
    """NVML's instantaneous board power (field NVML_FI_DEV_POWER_INSTANT) in watts, or -1 where the driver lacks it."""
    try:
        value = pynvml.nvmlDeviceGetFieldValues(handle, [pynvml.NVML_FI_DEV_POWER_INSTANT])[0]
        return value.value.uiVal / 1000.0 if value.nvmlReturn == 0 else -1.0
    except pynvml.NVMLError:
        return -1.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu", type=int, default=0, help="NVML index of the GPU to sample")
    parser.add_argument("--out", required=True, help="CSV file for the samples")
    parser.add_argument("--interval", type=float, default=0.002, help="seconds between samples")
    parser.add_argument("--busy", type=float, default=120.0, help="power (W) above which a sample counts as busy")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="command to run (after --)")
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        parser.error("no command given")

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu)
    samples = []
    start = time.perf_counter()
    process = subprocess.Popen(command)
    while process.poll() is None:
        sm = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
        mem = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        reasons = pynvml.nvmlDeviceGetCurrentClocksEventReasons(handle)
        instant = instant_power(handle)
        samples.append((time.perf_counter() - start, sm, mem, power, temp, reasons, instant))
        time.sleep(args.interval)
    pynvml.nvmlShutdown()

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("time,sm_mhz,mem_mhz,power_w,temp_c,reasons,instant_w\n")
        for t, sm, mem, power, temp, reasons, instant in samples:
            f.write(f"{t:.4f},{sm},{mem},{power:.1f},{temp},{reasons:#x},{instant:.1f}\n")
    busy = [s for s in samples if s[3] > args.busy]
    if busy:
        sm_median = statistics.median(s[1] for s in busy)
        mem_median = statistics.median(s[2] for s in busy)
        power_median = statistics.median(s[3] for s in busy)
        temp_max = max(s[4] for s in busy)
        bits = {}
        for s in busy:
            for bit in range(16):
                if s[5] & (1 << bit):
                    bits[1 << bit] = bits.get(1 << bit, 0) + 1
        reason_text = ", ".join(f"{bit:#x} {count / len(busy):.0%}" for bit, count in sorted(bits.items())) or "none"
        instants = sorted(s[6] for s in busy)
        print(f"sampler: instant power (NVML field 186) median {statistics.median(instants):.1f} W, "
              f"p95 {instants[int(0.95 * (len(instants) - 1))]:.1f} W, max {instants[-1]:.1f} W, "
              f"{len(set(instants))} distinct readings")
        print(f"sampler: {len(busy)} busy samples of {len(samples)} over {busy[-1][0] - busy[0][0]:.2f} s: "
              f"SM {sm_median:.0f} MHz, memory {mem_median:.0f} MHz, power {power_median:.1f} W (medians), max {temp_max} C, "
              f"reasons {reason_text}")
    else:
        print(f"sampler: no sample above {args.busy} W of {len(samples)}")
    return process.returncode


if __name__ == "__main__":
    sys.exit(main())
