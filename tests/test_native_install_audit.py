"""Compare audit identities with a real CMake build/install and reject altered ELF bytes."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from audit_native_install import installed_identity, sha


@unittest.skipUnless(sys.platform.startswith('linux') and all(shutil.which(x) for x in ('cc', 'cmake', 'readelf')),
                     'Linux C compiler, CMake and readelf are required')
class InstallIdentityContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='seedvr2-audit-test-')
        cls.root = Path(cls.tmp.name)
        (cls.root / 'main.c').write_text('int main(void) { return 0; }\n')
        (cls.root / 'CMakeLists.txt').write_text('''cmake_minimum_required(VERSION 3.25)
project(install_fixture C)
add_executable(fixture main.c)
set_target_properties(fixture PROPERTIES
  BUILD_RPATH "/fixture/build/path/with/padding/for/installation"
  INSTALL_RPATH "$ORIGIN/../lib:$ORIGIN")
install(TARGETS fixture RUNTIME DESTINATION bin)
''')
        commands = [['cmake', '-S', str(cls.root), '-B', str(cls.root / 'build')],
                    ['cmake', '--build', str(cls.root / 'build')],
                    ['cmake', '--install', str(cls.root / 'build'), '--prefix', str(cls.root / 'install')]]
        for command in commands:
            subprocess.run(command, capture_output=True, check=True)
        cls.original = cls.root / 'build/fixture'
        cls.installed = cls.root / 'install/bin/fixture'

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_cmake_install_reproduced_exactly(self):
        self.assertNotEqual(sha(self.original), sha(self.installed))
        result = installed_identity(self.original, self.installed, sha(self.original), 'lib')
        self.assertTrue(result['build_matches_frozen'])
        self.assertTrue(result['matches_reproduced_install'])

    def test_altered_installed_bytes_do_not_pass(self):
        target = self.root / 'tampered'
        shutil.copyfile(self.installed, target)
        with target.open('ab') as stream:
            stream.write(b'unreviewed change')
        self.assertFalse(installed_identity(self.original, target, sha(self.original), 'lib')['matches_reproduced_install'])

    def test_changed_build_cannot_inherit_old_numerical_evidence(self):
        self.assertFalse(installed_identity(self.original, self.installed, '0' * 64, 'lib')['build_matches_frozen'])


if __name__ == '__main__':
    unittest.main()
