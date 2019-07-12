from sciunit import scores
from sciunit import errors


class SmallerBetterScore(scores.FloatScore):
    _description = ('Score values where smaller is better')

    def __init__(self, score, *args, min_score, max_score, **kwargs):
        """
        Initialize the score. This class requires two extra mandatory
        keyword-only arguments.

        Parameters
        ----------
        score: Score value.
        min_score: Minimum possible score. This value is used to clip the
                   score value when computing norm_score. This is necessary
                   to avoid using very small/large values during coloring
                   which crashes sciunit. However, this value does not
                   affect the original score value in any way.
        max_score: Maximum possible score. This value is used to clip the
                   score value when computing norm_score. This is necessary
                   to avoid using very small/large values during coloring
                   which crashes sciunit. However, this value does not
                   affect the original score value in any way.
        """
        super().__init__(score, **kwargs)
        self.min_score = min_score
        self.max_score = max_score

    @property
    def norm_score(self):
        """Used for sorting. Smaller is better"""
        clipped = min(self.max_score, max(self.min_score, self.score))
        return (self.max_score - clipped) / (self.max_score - self.min_score)

    def color(self, value=None):
        """
        Always pass a normalized value to parent class'
        coloring method to avoid passing too large integers which
        crashes sciunit
        """
        if value is not None:
            self.score = value
        return super().color(self.norm_score)
