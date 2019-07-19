import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import datetime


class Timer(object):
    def __init__(self):
        self.start = datetime.datetime.now()

    def print(self, info="done"):
        end = datetime.datetime.now()
        print("\n[" + str(end - self.start) + "] " + info + "\n")

    def restart(self):
        self.start = datetime.datetime.now()


class RecommendSystem(object):
    def __init__(self, movies, edges, userNo):
        self.movies = movies
        self.edges = edges
        self.userNo = userNo
        self.movieNo = len(movies)

        self.split = int(0.9 * len(edges))

        self.F = self.get_F(self.edges[:self.split])
        self.calculate_W()
        self.RL = self.get_RL(self.F.T)

    def calculate_W(self):
        # adjacent matrix
        M = self.F
        Y = np.sum(self.F, axis=1)
        K = np.sum(self.F, axis=0)
        K[K != 0] = 1 / K[K != 0]

        M = M / np.reshape(Y, (self.userNo, 1))
        M = M * np.reshape(K, (1, self.movieNo))

        # calculate W matrix
        self.W = np.dot(M.T, M)

    def get_F(self, edges):
        F = np.zeros((self.userNo, self.movieNo))
        rows = tuple([r[0] for r in edges])
        cols = tuple([c[1] for c in edges])
        vals = [v[2] for v in edges]
        F[(rows, cols)] = vals
        return F

    def get_RL(self, F):
        F_ = np.dot(self.W, F)
        rec_list = np.argsort(-F_, axis=0)
        return rec_list

    def print_reclist(self, user):
        x = self.F[user - 1, :]
        f = np.argwhere(x == 1)
        p = np.reshape(f, (len(f),)).tolist()
        rl = np.reshape(self.RL[:, user - 1], (self.movieNo,)).tolist()
        for i in p:
            rl.remove(i)
        print("\nRecommend following movies to user {:d}:\n".format(user))
        for i in range(5):
            print(self.movies.iloc[rl[i]])

    def calculate_Rank(self):
        F = self.get_F(self.edges[self.split:]).T
        L = self.movieNo - np.sum(self.F.T, axis=0)
        rank = np.argsort(self.RL, axis=0)
        rank = rank / L
        return np.sum(rank[F == 1]) / np.sum(F)

    def calculate_ROC(self, step=50):
        TPR = [0 for i in range(step + 1)]
        FPR = [0 for i in range(step + 1)]
        GT = self.get_F(self.edges).T
        PR = self.get_RL(GT)

        for i in range(step + 1):
            rate = i / step
            PRM = get_PRM(PR, rate)

            TPR[i] = np.mean(tpr(GT, PRM))
            FPR[i] = np.mean(fpr(GT, PRM))

        return TPR, FPR


def tpr(GT, PR):
    rs = np.sum(np.logical_and(GT, PR), axis=0) / np.sum(GT, axis=0)
    return rs


def fpr(GT, PR):
    rs = (np.sum(PR, axis=0) - np.sum(np.logical_and(GT, PR), axis=0)) \
        / np.sum(np.logical_not(GT), axis=0)
    return rs


def get_PRM(rec_list, rate):
    PR = np.zeros_like(rec_list)
    J = np.zeros_like(rec_list)
    cols = J.shape[1]
    rows = J.shape[0]
    for i in range(cols):
        J[:, i] = i
    PR[rec_list[:int(rate * rows), :], J[:int(rate * rows), :]] = 1
    return PR


def auc_curve(TPR, FPR):
    roc_auc = auc(FPR, TPR)  # 计算auc的值

    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(FPR, TPR, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Data Analysis Homework by Ren Baojia 18171213373')
    plt.legend(loc="lower right")
    plt.show()


def initial_data(path):
    movies = pd.read_csv(path + 'movies.csv')
    ratings = pd.read_csv(path + 'ratings.csv')

    # initial parameters
    movieId = movies['movieId'].tolist()
    movieDic = dict(zip(movieId, [i for i in range(len(movieId))]))

    links = np.array(ratings[['userId', 'movieId', 'rating']], dtype=int)
    np.random.shuffle(links)
    userNo = np.max(links[:, 0])

    movieId = list(set(movieId).intersection(set(links[:, 1].tolist())))
    movieDic = dict(zip(movieId, [i for i in range(len(movieId))]))
    index = [movieDic[i] for i in movieId]
    movies = movies.iloc[index]

    edges = [(links[i, 0] - 1, movieDic[links[i, 1]],
              links[i, 2] >= 3) for i in range(len(links))]

    return movies, edges, userNo
