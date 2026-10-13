from typing import Optional
from statistics import mean, median

import networkx as nx
from ratingsystems import Game, Rating, RatingSystem
from ratingsystems.core.util import linear_regression_to_points\

from ratingsystems.rrs.model import PageRank


class RelativeRatingSystem(RatingSystem):

    name: str = "rrs"
    
    def __init__(
        self,
        alpha: float = 0.5,
        win_weight: Optional[int] = None,
        max_mov: Optional[int] = None,  # 30?
        location_bias: float = 2.5, # CFB: by points scored, home teams outscore equally matched away teams by about 2.5 points
        max_iter: int = 100000,
        baseline: bool = True,
        sink: bool = True,
    ):
        """
        Create a Relative Rating System.

        Args:
            alpha (float): alpha value for Page Rank algorithm (default: 0.5)
            win_weight (int): weight for each win in addition to margin of victory; if left empty, will use the median margin of victory for the dataset (default: None)
            max_mov (int): maximum value for margin of victory that can be applied for each win; if left empty, there is no maximum (default: None)
        """
        self.alpha = alpha
        self.win_weight = win_weight
        self.max_mov = max_mov
        self.location_bias = location_bias
        self.max_iter = max_iter
        self.baseline = baseline
        self.sink = sink

    def rate(self, games: list[Game]) -> Rating:
        wins_graph = nx.MultiDiGraph()
        losses_graph = nx.MultiDiGraph()
        wins_points = {}
        losses_points = {}
        teams = set()
        opponents = {}
        if self.win_weight is not None:
            win_weight = self.win_weight
        else:
            win_weight = median([abs(game.home_points - game.away_points) for game in games])
        for game in games:
            if isinstance(game, tuple):
                winner = game[0]
                loser = game[1]
                margin_of_victory = game[2]
            else:
                if game.home_points > game.away_points:
                    # Home team wins
                    margin_of_victory = game.home_points - game.away_points
                    if self.max_mov is not None:
                        margin_of_victory = min(margin_of_victory, self.max_mov)
                    if not game.neutral_site:
                        margin_of_victory -= self.location_bias
                    winner = game.home_team
                    loser = game.away_team
                elif game.away_points > game.home_points:
                    # Away team wins
                    margin_of_victory = game.away_points - game.home_points
                    if self.max_mov is not None:
                        margin_of_victory = min(margin_of_victory, self.max_mov)
                    if not game.neutral_site:
                        margin_of_victory += self.location_bias
                    winner = game.away_team
                    loser = game.home_team
                else:
                    print(game)
                    continue
            wins_graph.add_edge(loser, winner, weight=margin_of_victory + win_weight)
            losses_graph.add_edge(winner, loser, weight=margin_of_victory + win_weight)
            if winner not in wins_points:
                wins_points[winner] = 0
            wins_points[winner] += win_weight
            if loser not in losses_points:
                losses_points[loser] = 0
            losses_points[loser] += win_weight
            teams.add(winner)
            teams.add(loser)
            if winner not in opponents:
                opponents[winner] = []
            if loser not in opponents:
                opponents[loser] = []
            opponents[winner].append(loser)
            opponents[loser].append(winner)

        for team, points in wins_points.items():
            wins_graph.add_edge(team, team, weight=points)
        for team, points in losses_points.items():
            losses_graph.add_edge(team, team, weight=points)

        if self.baseline:
            wins_graph.add_node("baseline")
            losses_graph.add_node("baseline")

        if self.sink:
            wins_graph.add_node("sink")
            losses_graph.add_node("sink")
            for team in teams:
                wins_graph.add_edge(team, "sink", weight=max(win_weight, 1))
                losses_graph.add_edge(team, "sink", weight=max(win_weight, 1))

        wins_pagerank = nx.pagerank(wins_graph, alpha=self.alpha, weight="weight", max_iter=self.max_iter)
        losses_pagerank = nx.pagerank(losses_graph, alpha=self.alpha, weight="weight", max_iter=self.max_iter)

        wins_ratings = {}
        losses_ratings = {}
        if "exclude" in wins_pagerank:
            exclude_wins_rating = wins_pagerank["exclude"]
            exclude_losses_rating = losses_pagerank["exclude"]
            if self.baseline:
                exclude_wins_rating = (exclude_wins_rating - wins_pagerank["baseline"]) / (1 - wins_pagerank["baseline"] * (len(teams) + 1))
                exclude_losses_rating = (exclude_losses_rating - wins_pagerank["baseline"]) / (1 - wins_pagerank["baseline"] * (len(teams) + 1))
        for team in teams:
            if team == "exclude":
                continue
            wins_rating = wins_pagerank[team]
            losses_rating = losses_pagerank[team]
            if self.baseline:
                wins_rating = (wins_rating - wins_pagerank["baseline"]) / (1 - wins_pagerank["baseline"] * (len(teams) + 1))
                losses_rating = (losses_rating - losses_pagerank["baseline"]) / (1 - losses_pagerank["baseline"] * (len(teams) + 1))
            if self.sink:
                wins_rating = wins_rating / (1 - wins_pagerank["sink"])
                losses_rating = losses_rating / (1 - losses_pagerank["sink"])
            if "exclude" in wins_pagerank:
                wins_rating = wins_rating / (1 - exclude_wins_rating)
                losses_rating = losses_rating / (1 - exclude_losses_rating)
            wins_ratings[team] = PageRank(wins_rating)
            losses_ratings[team] = PageRank(-1 * losses_rating)

        wins_rating = (Rating(wins_ratings, games=games, _graph=wins_graph) * 10000) % "win"
        losses_rating = (Rating(losses_ratings, games=games, _graph=losses_graph) * 10000) % "loss"
        total_rating = wins_rating + losses_rating

        sos_rating = Rating({t: PageRank(sum([total_rating.get_value(o) for o in opponents[t]]) / len(opponents[t])) for t in teams}, name="sos", games=games)

        # return (total_rating % "rrs") << sos_rating
        return (linear_regression_to_points(total_rating, games) % "rrs") << sos_rating
        