#!/usr/bin/env python3
"""Retain explicit FMA operations in the project's Vulkan FP32 shaders.

GLSL precise protects arithmetic but glslang does not decorate GLSL.std.450.Fma.
NoContraction guards reassociation on the tested NVIDIA compiler. It does not
make core Vulkan Fma an IEEE single-rounding guarantee: runtime arithmetic
probes reject devices that discard the residual required by these kernels.
The build validates the resulting SPIR-V with spirv-val before embedding it.
"""
import argparse
from pathlib import Path
import struct


def protect(data):
    if len(data)<20 or len(data)%4:raise ValueError('Truncated SPIR-V')
    w=list(struct.unpack('<'+'I'*(len(data)//4),data))
    if w[0]!=0x07230203:raise ValueError('Invalid SPIR-V magic')
    instructions=[];offset=5
    while offset<len(w):
        size=w[offset]>>16;op=w[offset]&0xffff
        if not size or offset+size>len(w):raise ValueError('Invalid SPIR-V instruction length')
        instructions.append((offset,op,w[offset+1:offset+size]));offset+=size
    glsl=set();decorated=set();targets=set();insert=None
    for offset,op,args in instructions:
        if op==11: # OpExtInstImport
            if len(args)<2:raise ValueError('Invalid extended instruction import')
            name=struct.pack('<'+'I'*(len(args)-1),*args[1:]).split(b'\0',1)[0]
            if name==b'GLSL.std.450':glsl.add(args[0])
        elif op==71 and len(args)>=2 and args[1]==42:decorated.add(args[0])
        elif 19<=op<=39 and insert is None:insert=offset # Before type declarations.
    for _,op,args in instructions:
        if op==12 and len(args)>=4 and args[2] in glsl and args[3]==50: # Fma = 50.
            if len(args)!=7:raise ValueError('Invalid FMA arity')
            targets.add(args[1])
    missing=sorted(targets-decorated)
    if not missing:return data
    if insert is None:raise ValueError('SPIR-V type section missing')
    decoration=[v for result in missing for v in ((3<<16)|71,result,42)]
    w[insert:insert]=decoration
    return struct.pack('<'+'I'*len(w),*w)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('shader',type=Path);a=p.parse_args()
    data=a.shader.read_bytes();guarded=protect(data)
    if data!=guarded:a.shader.write_bytes(guarded)


if __name__=='__main__':main()
