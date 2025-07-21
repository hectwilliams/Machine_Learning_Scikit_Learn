from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import  StandardScaler
from matplotlib.lines import Line2D
import matplotlib.cm as cm

if __name__ == '__main__':

  mnist = datasets.fetch_openml('mnist_784', version=1)

  X, Y =  StandardScaler().fit_transform(mnist['data']) , mnist['target'].astype(int).to_numpy()

  clf_tsne = TSNE(n_components=2, n_jobs=8)
  print('training')

  X_train_reduce = clf_tsne.fit_transform(X)
  print('plotting')

  sc = plt.scatter(X_train_reduce[:,0], X_train_reduce[:,1] , s=1, c = Y/9,  cmap='hsv')

  plt.legend(
    handles = [ Line2D(range(1), range(1), color="white", marker='o', markerfacecolor=cm.get_cmap('hsv')( i / 10), label=f'class{i}' ) for i in range(10)]
  )
  plt.show()