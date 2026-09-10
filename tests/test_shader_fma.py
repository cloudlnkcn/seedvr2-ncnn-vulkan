"""Validate shader protection against the actual compiler and SPIR-V validator."""
import importlib.util
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('guard_shader_fma',ROOT/'tools/guard_shader_fma.py')
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)

class ShaderFmaTests(unittest.TestCase):
    def compile(self,root,expression):
        shader=root/'test.comp';shader.write_text('#version 450\nlayout(local_size_x=1) in;\nlayout(binding=0) buffer Data {float x[];};\nvoid main(){precise float y='+expression+';x[0]=y;}\n')
        output=root/'test.spv'
        subprocess.run(['glslangValidator','-V','--target-env','vulkan1.0',str(shader),'-o',str(output)],check=True,capture_output=True)
        return output

    def test_explicit_fma_is_protected_and_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            path=self.compile(Path(directory),'fma(x[0],x[1],x[2])');original=path.read_bytes();protected=guard.protect(original)
            self.assertEqual(protected[:20],original[:20]);self.assertEqual(guard.protect(protected),protected)
            path.write_bytes(protected)
            subprocess.run(['spirv-val','--target-env','vulkan1.0',str(path)],check=True,capture_output=True)
            text=subprocess.check_output(['spirv-dis',str(path)],text=True)
            ids=re.findall(r'(%\w+)\s*=\s*OpExtInst\s+%\w+\s+%\w+\s+Fma\b',text)
            self.assertTrue(ids)
            for result in ids:self.assertIn('OpDecorate '+result+' NoContraction',text)

    def test_unfused_expression_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            data=self.compile(Path(directory),'x[0]*x[1]+x[2]').read_bytes()
            self.assertEqual(guard.protect(data),data)

    def test_malformed_bytecode_is_rejected(self):
        for bad in (b'',b'bad!'+b'\0'*16,struct.pack('<6I',0x07230203,0x10000,0,20,0,0),struct.pack('<6I',0x07230203,0x10000,0,20,0,(4<<16)|11)):
            with self.subTest(data=bad),self.assertRaises(ValueError):guard.protect(bad)

if __name__=='__main__':unittest.main()
