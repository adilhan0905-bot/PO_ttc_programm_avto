import math
import unittest

from app.ttc import calculate_ttc, danger_level, estimate_distance_m


class TTCTests(unittest.TestCase):
    def test_distance_is_positive(self):
        distance = estimate_distance_m(1280, 120, "car", 90)
        self.assertIsNotNone(distance)
        self.assertGreater(distance, 0)

    def test_ttc_uses_vehicle_speed(self):
        ttc = calculate_ttc(10.0, 36.0)
        self.assertTrue(math.isclose(ttc, 1.0, rel_tol=1e-6))

    def test_temporal_approach_can_reduce_ttc(self):
        ttc = calculate_ttc(8.0, 10.0, previous_distance_m=10.0, fps=10.0)
        self.assertIsNotNone(ttc)
        self.assertLess(ttc, 2.0)

    def test_levels(self):
        self.assertEqual(danger_level(None, 2), "SAFE")
        self.assertEqual(danger_level(2.0, 2), "WARNING")
        self.assertEqual(danger_level(0.8, 2), "CRITICAL")


if __name__ == "__main__":
    unittest.main()
