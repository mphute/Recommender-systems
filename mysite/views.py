from django.http import HttpResponse
from django.shortcuts import render
import numpy as np
import pandas as pd
import collections

import tensorflow as tf



def mask(df, key, function):
  """Returns a filtered dataframe, by applying function to key"""
  return df[function(df[key])]

def flatten_cols(df):
  df.columns = [' '.join(col).strip() for col in df.columns.values]
  return df

pd.DataFrame.mask = mask
pd.DataFrame.flatten_cols = flatten_cols

movies = pd.read_csv("movies.csv" )
movies['title_orig']=movies['title']
movies['title'] = [x.lower() for x in movies['title']]
ratings = pd.read_csv("ratings.csv" )

correlation = pd.read_csv("corr_set.csv" )
df_books =  pd.read_csv("books.csv" )
df_books['title_orig'] = df_books['title']
df_books['title'] = [x.lower() for x in df_books['title']]



rated_movies = (ratings[["user_id", "movie_id"]]
                .groupby("user_id", as_index=False)
                .aggregate(lambda x: list(x)))
movies_ratings = movies.merge(
    ratings
    .groupby('movie_id', as_index=False)
    .agg({'rating': ['count', 'mean']})
    .flatten_cols(),
    on='movie_id')


DOT = 'dot'
COSINE = 'cosine'
def compute_scores(query_embedding, item_embeddings, measure=DOT):
  """Computes the scores of the candidates given a query.
  Args:
    query_embedding: a vector of shape [k], representing the query embedding.
    item_embeddings: a matrix of shape [N, k], such that row i is the embedding
      of item i.
    measure: a string specifying the similarity measure to be used. Can be
      either DOT or COSINE.
  Returns:
    scores: a vector of shape [N], such that scores[i] is the score of item i.
  """
  u = query_embedding
  V = item_embeddings
  if measure == COSINE:
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    u = u / np.linalg.norm(u)
  scores = u.dot(V.T)
  return scores


class CFModel(object):
  """Simple class that represents a collaborative filtering model"""
  def __init__(self, embedding_vars, loss, metrics=None):
    """Initializes a CFModel.
    Args:
      embedding_vars: A dictionary of tf.Variables.
      loss: A float Tensor. The loss to optimize.
      metrics: optional list of dictionaries of Tensors. The metrics in each
        dictionary will be plotted in a separate figure during training.
    """
    self._embedding_vars = embedding_vars
    self._loss = loss
    self._metrics = metrics
    self._embeddings = {k: None for k in embedding_vars}
    self._session = None

  @property
  def embeddings(self):
    """The embeddings dictionary."""
    return self._embeddings

  def loss(self):
      return self._loss



def movie_neighbors(model, title_substring, measure=DOT, k=10):
  # Search for movie ids that match the given substring.
  ids =  movies[movies['title'].str.contains(title_substring.lower())].index.values
  titles = movies.iloc[ids]['title'].values
  if len(titles) == 0:
    raise ValueError("Found no movies with title %s" % title_substring)
  
  movie_id = ids[0]
  scores = compute_scores(
      model.embeddings["movie_id"][movie_id], model.embeddings["movie_id"],
      measure)
  score_key = measure + ' score'
  df = pd.DataFrame({
      'score_key': list(scores),
      'titles': list(movies['title_orig']),
      'genres': list(movies['all_genres']),
      'images': list(movies['image_url'])
  })
  return (df.sort_values(['score_key'], ascending=False).head(k))


def get_top_10(title_substring):
    #title_substring = "Pride and Prejudice"

    ids =  df_books[df_books['title'].str.contains(title_substring.lower())].index.values
    titles = df_books.iloc[ids]['title'].values
    if len(titles) == 0:
        raise ValueError("Found no movies with title %s" % title_substring)
    
    bk_id = ids[0]
    #score_key = measure + ' score'
    df = pd.DataFrame({
      'score': correlation.iloc[bk_id],
      'titles': list(df_books['title_orig']),
      'author': list(df_books['author']),
      'year': list(df_books['year']),
      'images' : list(df_books['image_url'])
      })
    k=10
    return(df.sort_values(['score'], ascending=False).head(k))
    
soft_loss2 = np.load("sfloss.npy" , allow_pickle=True)

sfloss = tf.convert_to_tensor(soft_loss2, dtype=np.float32)
  
emb = np.load('movie_embeddings.npy')
emb_dict = dict()
emb_dict['movie_id'] = emb

softmodel = CFModel(emb_dict,sfloss)
softmodel.embeddings['movie_id'] = emb


##############################


def bookrec(request):
        
        title_request = request.GET["inputBook"]
        title_substring = title_request

        df10 = get_top_10(title_substring)
        print(df10['titles'])
        print(df10['images'])
        context = {'titles' : df10['titles'] , 'author' : df10['author'] , 'year' : df10['year'] , 'images' : df10['images'] , 'input':title_request}
        return render(request,'bookrec.html',context=context)


def bookcover(request):
        context = {'foo': 'bar'}
        return render(request,'bookcover.html',context = context)

def movierec(request):
        


        title_request = request.GET["inputMovie"]
        title_substring = title_request

        df10 = movie_neighbors(softmodel, title_substring, COSINE)
        print(df10['titles'])
        print(df10['images'])
        context = {'titles' : list(df10['titles']) , 'genres' : list(df10['genres']) ,'input':title_request, 'images' : df10['images']}
        return render(request,'movierec.html',context = context)

def moviecover(request):
        context = {'foo': 'bar'}
        return render(request,'moviecover.html',context = context)

