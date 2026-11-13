#!/bin/bash
# Print GPU utilisation and any other compute processes; used before each measurement to document conditions.
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw,clocks.sm --format=csv,noheader | sed 's/^/gpu util,memused,power,smclk: /'
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | cut -c1-110
