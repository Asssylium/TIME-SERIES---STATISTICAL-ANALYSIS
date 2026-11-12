import unittest
from unittest.mock import patch

from src import native_runtime


class NativeRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, native_runtime, "_openmp_handle", native_runtime._openmp_handle)
        native_runtime._openmp_handle = None

    def test_linux_preloads_once_with_global_symbols(self):
        with patch.object(native_runtime.sys, "platform", "linux"), \
                patch.object(native_runtime.Path, "is_file", return_value=True), \
                patch.object(native_runtime.ctypes, "CDLL") as load:
            native_runtime.load_bundled_openmp()
            native_runtime.load_bundled_openmp()
            load.assert_called_once()
            self.assertTrue(load.call_args.args[0].endswith("libgomp.so.1"))
            self.assertEqual(load.call_args.kwargs["mode"], native_runtime.ctypes.RTLD_GLOBAL)

    def test_local_environment_without_bundle_is_unchanged(self):
        with patch.object(native_runtime.sys, "platform", "linux"), \
                patch.object(native_runtime.Path, "is_file", return_value=False), \
                patch.object(native_runtime.ctypes, "CDLL") as load:
            native_runtime.load_bundled_openmp()
            load.assert_not_called()

    def test_windows_does_not_load_linux_library(self):
        with patch.object(native_runtime.sys, "platform", "win32"), \
                patch.object(native_runtime.ctypes, "CDLL") as load:
            native_runtime.load_bundled_openmp()
            load.assert_not_called()
