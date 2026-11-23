#!/bin/bash
# sum.sh <en output>: route product variant ms W MHz, per round
awk '/energy:/ {for(i=1;i<=NF;i++){if($i=="routing,"&&$(i-2)=="ms")ms=$(i-3); if($i=="W,")w=$(i-1); if($i=="MHz")mhz=$(i-1)} printf "%-3s %-5s %-3s %-12s %s %s W %s MHz\n",$2,$3,$4,$5,ms,w,mhz}' "$@"
