import pandas as pd
from socceraction.data.statsbomb import StatsBombLoader
import socceraction.spadl as spadl
import socceraction.xthreat as xthreat
import statsbombpy
import sys
import os


os.chdir(
    "/Users/riccardobergamin/Documents/GitHub/BCOS/examples/statbomb"
)
sys.path.append(
    "/Users/riccardobergamin/Documents/GitHub/BCOS/examples/statbomb/scripts"
)

from plot import plot_xt

# 1. Load a set of actions to train the model on
SBL = StatsBombLoader()
df_games = SBL.games(competition_id=9, season_id=281)
dataset = [
    {
        **game,
        'actions': spadl.statsbomb.convert_to_actions(
            events=SBL.events(game['game_id']),
            home_team_id=game['home_team_id']
        )
    }
    for game in df_games.to_dict(orient='records')
]

# 2. Convert direction of play + add names
df_actions_ltr = pd.concat([
  spadl.play_left_to_right(game['actions'], game['home_team_id'])
  for game in dataset
])
df_actions_ltr = spadl.add_names(df_actions_ltr)

# 3. Train xT model with 16 x 12 grid
xTModel = xthreat.ExpectedThreat(l=16, w=12)
xTModel.fit(df_actions_ltr)

# Estrai la matrice
xt_matrix = xTModel.heatmaps

xT = xt_matrix[-1]

pd.DataFrame(xT).to_csv("output/xT_grid.csv", index=False, header=None)


# Esegui la funzione passandogli il tuo xT
plot_xt(xT)

