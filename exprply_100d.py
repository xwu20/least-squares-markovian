#code for Least Squares Regression with Markovian Data: Fundamental Limits and Algorithms
#Authors: Guy Bresler (MIT), Prateek Jain (MSR), Dheeraj Nagaraj (MIT), Praneeth Netrapalli (MSR), Xian Wu (Stanford)
#Published at NeurIPS 2020

import numpy as np
from sklearn.preprocessing import normalize
from numpy.linalg import norm
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

d=100
n_list=[10, 100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 200000, 400000, 600000, 800000, 10000000, 20000000, 40000000, 100000000]
sig=0.001
eps_list=[.01]
eps=.01
numtest=10
err_sgd=np.zeros((len(n_list),numtest))
err_opt=np.zeros((len(n_list),numtest))
err_pr_batchavg=np.zeros((len(n_list),numtest))
err_sgdd=np.zeros((len(n_list),numtest))

for nidx, n in enumerate(n_list):
    print("in new outer loop")
    X = np.zeros((n, d))
    y = np.zeros(n)
    avg_point = int(n / 2)
    for test in range(numtest):
        #print(str(eps) + " " + str(test))
        G = np.random.normal(0, 1, (n, d))
        G = normalize(G, axis=1, norm='l2')

        wopt = np.random.normal(0, 1, d)
        wopt = wopt / np.linalg.norm(wopt)
        tau = np.random.normal(0, sig, n)
        for i in range(n):
            if i == 0:
                X[i] = np.random.normal(0, 1, (1, d))
                X[i] = X[i] / np.linalg.norm(X[i])
            else:
                X[i] = np.sqrt(1 - eps * eps) * X[i - 1] + eps * G[i - 1]
            y[i] = np.dot(X[i], wopt) + tau[i]
        #print("Data generated")
        B = int(np.ceil((1 / eps) ** (2)))

        w0 = np.random.normal(0, 1, d)
        w = w0 / np.linalg.norm(w0)
        eta = .2
        w_sgd = np.zeros(d)
        for i in range(n):
            wold = w
            w = w - eta * (np.dot(X[i], w) - y[i]) * X[i].T
            if i >= avg_point:
                w_sgd = w_sgd * ((i - avg_point) / (i - avg_point + 1)) + w / (i - avg_point + 1)
            #if (i % 100000 == 0):
                #print("SGD itr:" + str(i))
        err_sgd[nidx, test] = norm(w_sgd - wopt) ** 2

        w = w0 / np.linalg.norm(w0)
        w_pr_batchavg = np.zeros(d)
        avg_point_B = int(np.floor(avg_point / B))
        for t in range(int(np.floor(n / B))):
            for tau in range(B):
                i = np.random.randint(0, B) + B * t
                w = w - eta * (np.dot(X[i], w) - y[i]) * X[i].T
            if t >= avg_point_B:
                w_pr_batchavg = w_pr_batchavg * ((t - avg_point_B) / (t - avg_point_B + 1)) + w / (t - avg_point_B + 1)
            #if ((t * B) % 100000 == 0):
                #print("ER itr:" + str(t))
        num_batch = int(np.floor(n / B))
        for tau in range(n - B * num_batch):
            i = np.random.randint(0, n - B * num_batch) + B * num_batch
            w = w - eta * (np.dot(X[i], w) - y[i]) * X[i].T
        w_pr_batchavg = w_pr_batchavg * ((num_batch - avg_point_B) / (num_batch - avg_point_B + 1)) + w / (
                    num_batch - avg_point_B + 1)
        err_pr_batchavg[nidx, test] = norm(w_pr_batchavg - wopt) ** 2

        w0 = np.random.normal(0, 1, d)
        w = w0 / np.linalg.norm(w0)
        eta = .2
        w_sgdd = np.zeros(d)
        DD = int(1 / eps ** 2 * np.log(d))
        for i in range(n):
            wold = w
            if (i % DD == 0):
                w = w - eta * (np.dot(X[i], w) - y[i]) * X[i].T
            if i >= avg_point:
                w_sgdd = w_sgdd * ((i - avg_point) / (i - avg_point + 1)) + w / (i - avg_point + 1)
            #if (i % 100000 == 0):
                #print("SGDD itr:" + str(i))
        err_sgdd[nidx, test] = norm(w_sgdd - wopt) ** 2

Err_ba=np.mean(err_pr_batchavg,axis=1)
Err_sgd=np.mean(err_sgd,axis=1)
Err_sgdd=np.mean(err_sgdd,axis=1)


npoints=int(n)
plt.xscale("log")
plt.yscale("log")
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title(r'Error wrt $T$ (d=100,$\epsilon$=1e-2,$\sigma$=1e-3)',fontsize=18)
plt.ylabel(r'$\|\|w-w^*\|\|^2$',fontsize=18)
plt.xlabel(r'T',fontsize=18)
plt.plot(n_list,Err_sgd,'k',label="SGD")
plt.plot(n_list,Err_sgdd,'r',label="SGD-DD")
plt.plot(n_list,Err_ba,'b',label="SGD-ER")
plt.plot([d/eps**2, d/eps**2*np.log(d)],[3e-9, 2],'k--',label=r'$d/\epsilon^2$')
plt.plot([d/eps, d/eps*np.sqrt(d)],[3e-9, 2],'b--',label=r'$d/\epsilon$')
plt.legend(fontsize=14)
plt.savefig('exprply100d.png',bbox_inches='tight')

