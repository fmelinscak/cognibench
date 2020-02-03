import unittest
from cognibench.scores import HigherBetterScore, LowerBetterScore


class TestHigherBetterScore(unittest.TestCase):
    def test_norm_score(self):
        s1 = HigherBetterScore(40.0, min_score=0, max_score=100)
        s2 = HigherBetterScore(60.0, min_score=0, max_score=100)
        self.assertTrue(s1.norm_score < s2.norm_score)

    def test_score_out_of_limits(self):
        s2 = HigherBetterScore(60.0, min_score=0, max_score=50)
        self.assertTrue(s2.norm_score == 1.0)


class TestLowerBetterScore(unittest.TestCase):
    def test_norm_score(self):
        s1 = LowerBetterScore(40.0, min_score=0, max_score=100)
        s2 = LowerBetterScore(60.0, min_score=0, max_score=100)
        self.assertTrue(s1.norm_score > s2.norm_score)

    def test_score_out_of_limits(self):
        s2 = LowerBetterScore(60.0, min_score=0, max_score=50)
        self.assertTrue(s2.norm_score == 0.0)
