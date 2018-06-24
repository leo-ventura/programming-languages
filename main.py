# importing necessary modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# reading the csv from github
df = pd.read_csv('https://raw.githubusercontent.com/leo-ventura/programming-languages/master/data.csv')

# temos que separar as linguagens de programação dos usuários (linguages como nosso x e usuarios como y)
features = ['assembly','batchfile','c','c#','c++','clojure','coffeescript','css','elixir','emacs lisp','go','haskell','html','java','javascript','jupyter notebook','kotlin','lua','matlab','objective-c','objective-c++','ocaml','perl','php','powershell','purebasic','python','rascal','ruby','rust','scala','shell','swift','tex','typescript','vim script','vue']

x = df.loc[:,features].values

# a ser usado no plot
marker = itertools.cycle(('X', 'o', '*'))

# criando vetor de soma 1
X_sum1 = np.copy(x)
eps = 10**-6

for i in range(0, X_sum1.shape[0]):
    if (np.sum(X_sum1[i,:]) > eps):
        X_sum1[i,:] = X_sum1[i,:]/np.sum(X_sum1[i,:])
    else:
        X_sum1[i,:] = 0 * X_sum1[i,:] 

cov_mat = np.cov(X_sum1.T)

#print("Covariant matrix:", cov_mat)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

# Construindo matriz de reducao de dimensionalidade
P = eig_vecs[:,0:2]
points_PCA_2d = X_sum1 @ P                      # Projetando na base das direcoes principais

# Fazendo o mesmo, só que para 3 dimensoes (melhores resultados)
P = eig_vecs[:,0:3]
points_PCA_3d = X_sum1 @ P                      # Projetando na base das direcoes principais

# Clusterizando

points_clustered_2d = [np.zeros(2)]*len(features)
points_clustered_3d = [np.zeros(3)]*len(features)

for i in range(X_sum1.shape[0]):
    maxElem = np.argmax(X_sum1[i,:])
    points_clustered_2d[maxElem] = np.vstack( (points_clustered_2d[maxElem], points_PCA_2d[i,:]) )
    points_clustered_3d[maxElem] = np.vstack( (points_clustered_3d[maxElem], points_PCA_3d[i,:]) )


####### Ploting #######
# 2d


fig = plt.figure(figsize=(12,10))


for i in range( len(features) ):
    if(len(points_clustered_2d[i].shape) > 1):
      plt.plot(points_clustered_2d[i][:,0], points_clustered_2d[i][:,1], marker=next(marker) , label = features[i], color = np.random.rand(3) , markersize=10, linestyle='')

fig.legend()

# 3d 
fig = plt.figure(figsize=(15,10))
ax  = fig.add_subplot(111, projection='3d')

for i in range( len(features) ):
    if(len(points_clustered_3d[i].shape) > 1):
      ax.scatter(points_clustered_3d[i][:,0], points_clustered_3d[i][:,1], points_clustered_3d[i][:,2], marker=next(marker), label = features[i], color = np.random.rand(3), s=21)

ax.set_xlim(-0.3,0.72)
ax.set_ylim(-0.4,0.8)
ax.set_zlim(-0.8,0.5)
fig.legend()

# Plot entre duas linguagens no 2d

fig = plt.figure(figsize=(12,10))


# index da primeira linguagem
idx_primeira_linguagem = features.index('javascript')
idx_segunda_linguagem  = features.index('html')

plt.plot(points_clustered_2d[idx_primeira_linguagem][:,0], points_clustered_2d[idx_primeira_linguagem][:,1], marker=next(marker) , label = features[idx_primeira_linguagem], color = 'turquoise' , markersize=10, linestyle='')
plt.plot(points_clustered_2d[idx_segunda_linguagem][:,0], points_clustered_2d[idx_segunda_linguagem][:,1], marker=next(marker) , label = features[idx_segunda_linguagem], color = 'fuchsia' , markersize=10, linestyle='')

fig.legend()

# duas linguagens de programacao em 3d 

fig = plt.figure(figsize=(15,10))
ax  = fig.add_subplot(111, projection='3d')


idx_primeira_linguagem = features.index('javascript')
idx_segunda_linguagem  = features.index('python')
idx_terceira_linguagem = features.index('html')

ax.scatter(points_clustered_3d[idx_primeira_linguagem][:,0], points_clustered_3d[idx_primeira_linguagem][:,1], points_clustered_3d[idx_primeira_linguagem][:,2], marker=next(marker), label = features[idx_primeira_linguagem], color = 'blue', s=21)
ax.scatter(points_clustered_3d[idx_segunda_linguagem][:,0], points_clustered_3d[idx_segunda_linguagem][:,1], points_clustered_3d[idx_segunda_linguagem][:,2], marker=next(marker), label = features[idx_segunda_linguagem], color = 'green', s=21)
ax.scatter(points_clustered_3d[idx_terceira_linguagem][:,0], points_clustered_3d[idx_terceira_linguagem][:,1], points_clustered_3d[idx_terceira_linguagem][:,2], marker=next(marker), label = features[idx_terceira_linguagem], color = 'fuchsia', s=21)

ax.set_xlim(-0.3,0.72)
ax.set_ylim(-0.4,0.8)
ax.set_zlim(-0.8,0.5)
fig.legend()