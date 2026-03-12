import math
import networkx as nx
import scipy.stats as st

from ratingsystems import Prediction, Predictor, Rating


class RelativeRatingSystemMarkovChainPredictor(Predictor):

    name: str = "rrs"

    def predict(self, team: str, opponent: str) -> Prediction:
        wins_pagerank = nx.pagerank(self.rating.win._raw._graph, personalization={team: 0.5, opponent: 0.5}, alpha=0.5, weight="weight", max_iter=10000, nstart={team: 0.5, opponent: 0.5})
        losses_pagerank = nx.pagerank(self.rating.loss._raw._graph, personalization={team: 0.5, opponent: 0.5}, alpha=0.5, weight="weight", max_iter=10000, nstart={team: 0.5, opponent: 0.5})

        team_matchup_rating = wins_pagerank[team] - losses_pagerank[team]
        opponent_matchup_rating = wins_pagerank[opponent] - losses_pagerank[opponent]
        # line = (team_matchup_rating - opponent_matchup_rating) * 100

        # odds = team_matchup_rating / (team_matchup_rating + opponent_matchup_rating)
        odds = 1 / (1 + pow(math.e, 26 * (opponent_matchup_rating - team_matchup_rating)))

        return Prediction(
            team,
            opponent,
            odds=odds,
            line=st.norm.ppf(odds) * self.rating.confidence_interval,
        )
