import unittest

from scripts.main_phase_anchor import (
    main_anchor_reliability_from_fit_audit,
    main_prenudge_cfg,
)


class MainAnchorPolicyTests(unittest.TestCase):
    def test_unreliable_main_anchor_is_not_default_hard_stop(self):
        cfg = main_prenudge_cfg({}, {}, {"reps": 20, "samples": 2000})
        self.assertFalse(cfg["fail_unresolved_main"])

    def test_high_main_only_rwp_marks_anchor_unreliable(self):
        audit = {
            "triggered": True,
            "reason": "hard_rwp_trigger",
            "rwp": 57.1,
            "points": 3669,
            "weighted_peak_support": 0.506,
        }
        reliable, reason = main_anchor_reliability_from_fit_audit(
            audit,
            {
                "require_reliable_main_rwp_max": 25.0,
                "require_reliable_main_peak_support": 0.55,
            },
        )
        self.assertFalse(reliable)
        self.assertIn("hard_rwp_trigger", reason)


if __name__ == "__main__":
    unittest.main()