#!/bin/bash
# nvrtc_check.sh kernel.cu [-DNAME=value ...]: the kernel through the provider's NVRTC (its options), then ptxas for sm_120:
# registers, spill stores and the local-memory accesses between the first and last MMA. env: NVRTC_DIR (libnvrtc.so.13 and builtins).
src=$1; shift
tmp=${TMPDIR:-.}/t.cu; : > $tmp
for d in "$@"; do m=${d#-D}; echo "#define ${m%%=*} ${m#*=}" >> $tmp; done
cat $src >> $tmp
LD_LIBRARY_PATH=${NVRTC_DIR:?directory with libnvrtc.so.13} python3 "$(dirname "$0")/nvrtc_ptx.py" $tmp ${TMPDIR:-.}/t.ptx || exit 1
ptxas -arch=sm_120 -v ${TMPDIR:-.}/t.ptx -o ${TMPDIR:-.}/t.cubin 2>&1 | grep -oE "Used [0-9]+ registers|[0-9]+ bytes spill stores" | tr '\n' ' '
cuobjdump -sass ${TMPDIR:-.}/t.cubin | grep -E "^\s+/\*[0-9a-f]+\*/" | awk '{print $2}' | sed 's/\..*//' | awk '{a[NR]=$1} END{first=0;last=0; for(i=1;i<=NR;i++) if(a[i]=="QMMA"){ if(!first)first=i; last=i}; c=0; for(i=first;i<=last;i++) if(a[i]=="STL"||a[i]=="LDL") c++; print "spills in MMA span:", c}'
