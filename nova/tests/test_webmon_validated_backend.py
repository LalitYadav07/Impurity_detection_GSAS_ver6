import unittest
from unittest.mock import patch
from radar_pd_nova.webmon.validated_backend import corrected_backend,active_identity,VERSION

class BackendTests(unittest.TestCase):
 def test_version_pinning_and_restore(self):
  from GSASII import GSASIIstrMath as M
  original=M.MagStructureFactor2
  with corrected_backend() as identity:
   self.assertEqual(identity['version'],VERSION);self.assertIsNot(M.MagStructureFactor2,original)
   self.assertEqual(active_identity(),identity)
   with self.assertRaises(RuntimeError):
    with corrected_backend():pass
  self.assertIs(M.MagStructureFactor2,original);self.assertIsNone(active_identity())
  with self.assertRaises(ValueError):
   with corrected_backend():raise ValueError('worker failure')
  self.assertIs(M.MagStructureFactor2,original)
 def test_wrong_native_version_refused(self):
  with patch('radar_pd_nova.webmon.validated_backend.PINNED_STRMATH','wrong'),patch('radar_pd_nova.webmon.validated_backend.PINNED_LINUX_STRMATH','wrong'):
   with self.assertRaises(RuntimeError):
    with corrected_backend():pass
  with corrected_backend():pass
 def test_single_crystal_refused(self):
  from GSASII import GSASIIstrMath as M
  with corrected_backend():
   with self.assertRaises(ValueError):M.MagStructureFactor2({},None,':0:','0::',{}, {':0:histType':'SNC'}, {})

if __name__=='__main__':unittest.main()
