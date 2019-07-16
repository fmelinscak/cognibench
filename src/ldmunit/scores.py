from sciunit import scores
from sciunit import errors


class SmallerBetterScore(scores.FloatScore):
    """
    SmallerBetterScore is a score type where smaller values are better than
    larger values, similar to an error function. This property is used by
    sciunit library when sorting or color coding the scores.

    In addition to the above primary functionality, this class serves a
    secondary purpose: If python int values that are larger than 64 bit
    signed integers are used as scores, sciunit coloring code crashes since
    due to conversion errors. This class allows defining minimum and maximum
    ranges that are used for clipping the score only during the coloring part
    so that too large values are never passed to sciunit coloring code.

    Minimum and maximum scores also define the coloring scheme: All the scores
    less than or equal to the minimum value will be displayed using the maximum
    green color and all the scores greater than or equal to the maximum value
    will be displayed using the maximum red color. All the values in between
    get assigned a color using linear interpolation between red and green.
    """
    _description = ('Score values where smaller is better')

    def __init__(self, score, *args, min_score, max_score, **kwargs):
        """
        Initialize the score. This class requires two extra mandatory
        keyword-only arguments.

        Parameters
        ----------
        score : float
            Score value.

        min_score : float
            Minimum possible score. This value is used to clip the
            score value when computing norm_score. This is necessary
            to avoid using very small/large values during coloring
            which crashes sciunit. However, this value does not
            affect the original score value in any way.

        max_score : float
            Maximum possible score. This value is used to clip the
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
        """
        Used for sorting. Smaller is better.

        Returns
        -------
        float
            Score value normalized to 0-1 range computed by clipping self.score to the
            min/max range and then transforming to a value in [0, 1].
        """
        clipped = min(self.max_score, max(self.min_score, self.score))
        return (self.max_score - clipped) / (self.max_score - self.min_score)

    def color(self, value=None):
        """
        Ensure that a normalized value is passed to parent class' color method which
        does the real work.

        Parameters
        ----------
        value : float
            Score value to color. If None, function uses self.score

        See Also
        --------
        scores.FloatScore.color
        """
        if value is not None:
            self.score = value
        return super().color(self.norm_score)
