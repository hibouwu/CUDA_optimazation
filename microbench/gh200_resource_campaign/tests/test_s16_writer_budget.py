"""Small CPU writer protocol, real byte/hash and bounded partial-budget tests."""
import hashlib,json,subprocess,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class WriterBudgetTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tmp=tempfile.TemporaryDirectory();cls.binary=Path(cls.tmp.name)/'writer'
  result=subprocess.run(['g++','-std=c++17','-O3',str(ROOT/'probes/feasibility/s16_writer_budget_v1.cpp'),'-o',str(cls.binary)],capture_output=True,text=True)
  assert result.returncode==0,result.stderr
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup()
 def test_full24_small_shape_hash_and_max_arithmetic(self):
  folder=Path(self.tmp.name)/'full';r=subprocess.run([str(self.binary),str(folder),'1','30'],capture_output=True,text=True,timeout=30);self.assertEqual(r.returncode,0,r.stderr);rows=[json.loads(x) for x in r.stdout.splitlines()];self.assertEqual(len(rows),25);summary=rows[-1];self.assertTrue(summary['full_shape_completed']);self.assertEqual(summary['files'],24);self.assertFalse(summary['GPU_execution']);self.assertEqual(summary['expected_full_bytes'],11341024);self.assertEqual((summary['expected_full_bytes']-128)*396+128,4490994944)
  for row in rows[:-1]:
   data=(folder/row['path']).read_bytes();self.assertEqual(len(data),row['bytes']);self.assertEqual(hashlib.sha256(data).hexdigest(),row['sha256'])
  self.assertEqual(sum(x['bytes'] for x in rows[:-1]),summary['completed_bytes'])
 def test_tiny_budget_retains_prefix_and_exits_before_complete(self):
  folder=Path(self.tmp.name)/'partial';r=subprocess.run([str(self.binary),str(folder),'1','0.000000001'],capture_output=True,text=True,timeout=30);self.assertEqual(r.returncode,3);rows=[json.loads(x) for x in r.stdout.splitlines()];self.assertFalse(rows[-1]['full_shape_completed']);self.assertEqual(rows[-1]['status'],'writer_budget_exceeded');self.assertLess(rows[-1]['files'],24);self.assertGreater(rows[-1]['writer_seconds'],rows[-1]['budget_seconds'])
 def test_no_overwrite_invalid_blocks_and_nonfinite_budget(self):
  for index,(blocks,seconds) in enumerate([('0','30'),('1025','30'),('1','nan'),('1','inf'),('1','31')]):
   folder=Path(self.tmp.name)/('bad'+str(index));r=subprocess.run([str(self.binary),str(folder),blocks,seconds],capture_output=True,text=True);self.assertNotEqual(r.returncode,0);self.assertFalse(folder.exists())
  folder=Path(self.tmp.name)/'existing';folder.mkdir();r=subprocess.run([str(self.binary),str(folder),'1','30'],capture_output=True,text=True);self.assertNotEqual(r.returncode,0)
if __name__=='__main__':unittest.main()
