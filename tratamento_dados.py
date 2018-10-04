import pandas as pd
import numpy as np
from ast import literal_eval

md = pd.read_csv('dataset/movies_metadata.csv', low_memory=False)
md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(
    lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = md[md['vote_average'].notnull()]['vote_average'].astype('int')
C = vote_averages.mean()

m = vote_counts.quantile(0.95)
md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(
    lambda x: str(x).split('-')[0] if x != np.nan else np.nan)

qualified = md[(md['vote_count'] >= m) & (md['vote_count'].notnull()) & (md['vote_average'].notnull())][
    ['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]
qualified['vote_count'] = qualified['vote_count'].astype('int')
qualified['vote_average'] = qualified['vote_average'].astype('int')

df = pd.Series(qualified['genres']).apply(frozenset).to_frame(name='genre')
for genre in frozenset.union(*df.genre):
    df[genre] = df.apply(lambda _: int(genre in _.genre), axis=1)

dataset = pd.concat([qualified, df.iloc[:, 1:]], axis=1)
dataset.to_csv('dataset.csv')
