from ratingsystems import Stat


class PageRank(Stat):

    def __init__(self, value: float):
        super().__init__(value)

    def formatted(self, precision: int = 0) -> str:
        return f"%.{precision}f" % round(self.value, precision)
