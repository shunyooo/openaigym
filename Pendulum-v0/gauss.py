import numpy as np
import matplotlib.pyplot as plt

#ガウス一次元確率密度を返す関数
def f(x,mu = 0,sigma = 1):
    return np.exp((-(x-mu)**2)/(2.0*(sigma**2))) / (np.sqrt(2*np.pi)*sigma)

# 平均と分散のテスト用リスト
mus = np.arange(-20,20,10)
sigmas = np.arange(1,10,2)

x = np.arange(-20,20,0.1)

# 出力時のFigureの幅、高さ指定
w = len(sigmas)*3
h = len(mus)*3

# サブプロット（figure内に複数のグラフ(ax)）を入れる方式。先に座標を取るやり方。
# http://ailaby.com/matplotlib_fig/ 参考
fig, axs = plt.subplots(len(mus), len(sigmas), figsize=(w, h),sharex=True, sharey=True)

for i in range(len(mus)):
	for j in range(len(sigmas)):
		axs[i,j].plot(x,f(x,mu=mus[i],sigma=sigmas[j]))
		axs[i,j].set_title("平均:{}, 分散{}".format(mus[i],sigmas[j]))
    
# fig.tight_layout()  # タイトルとラベルが被るのを解消
plt.show()

