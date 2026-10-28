import io
import unittest
from proofgame.epoch_progress import EpochProgress, duration


class EpochProgressTests(unittest.TestCase):
    def test_eta_includes_full_batch_time_and_finishes_at_zero(self):
        now=[100.]
        output=io.StringIO()
        bar=EpochProgress(2,'train',100,stream=output,clock=lambda:now[0])
        self.assertIsNone(bar.show()['eta_seconds'])
        now[0]+=20
        data=bar.show(10)
        self.assertEqual(data['eta_seconds'],180.)
        self.assertEqual(data['percent'],10.)
        now[0]+=180
        self.assertEqual(bar.show(100)['eta_seconds'],0.)
        self.assertIn('Epoch 2 train',output.getvalue())
        self.assertIn('train ETA ~3m 00s',output.getvalue())
        self.assertNotIn('\r',output.getvalue())

    def test_resume_excludes_previous_work_and_downtime(self):
        now=[5000.]
        bar=EpochProgress(2,'eval',100,60,stream=io.StringIO(),clock=lambda:now[0])
        self.assertIsNone(bar.show()['eta_seconds'])
        now[0]+=40
        self.assertEqual(bar.show(80)['eta_seconds'],40.)
        with self.assertRaises(ValueError):
            bar.show(79)

    def test_redirected_heartbeat_is_quiet_and_empty_phase_is_complete(self):
        output=io.StringIO()
        bar=EpochProgress(0,'baseline',0,stream=output)
        self.assertEqual(bar.show(heartbeat=True)['percent'],100.)
        self.assertEqual(output.getvalue(),'')
        self.assertEqual(bar.show()['eta_seconds'],0.)
        self.assertEqual(duration(3660),'1h 01m')


if __name__=='__main__':
    unittest.main()
