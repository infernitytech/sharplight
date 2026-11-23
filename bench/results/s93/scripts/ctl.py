import sys,re
# decode Volta+ control bits from cuobjdump -sass: instr line followed by "/* 0x... */" upper word
lines=open(sys.argv[1]).read().split('\n')
lo,hi=int(sys.argv[2],16),int(sys.argv[3],16)
i=0
while i<len(lines):
    m=re.search(r'/\*([0-9a-f]{4})\*/\s+(.*?);\s*/\* (0x[0-9a-f]+) \*/',lines[i])
    if m:
        addr=int(m.group(1),16)
        h=re.search(r'/\* (0x[0-9a-f]+) \*/',lines[i+1])
        hiw=int(h.group(1),16)
        ctl=hiw>>41
        stall=ctl&0xf; yld=(ctl>>4)&1; wb=(ctl>>5)&7; rb=(ctl>>8)&7; wait=(ctl>>11)&0x3f
        if lo<=addr<=hi:
            print(f"{addr:04x} st{stall:2d} wb{wb if wb!=7 else '-'} rb{rb if rb!=7 else '-'} wait{wait:06b} {m.group(2).strip()}")
        i+=2
    else: i+=1
