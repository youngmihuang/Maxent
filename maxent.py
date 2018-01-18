# coding: utf-8
from collections import defaultdict
import math

class MaxEnt(object):
    def __init__(self):
        self.feats = defaultdict(int)
        self.trainset = [] # 訓練集
        self.labels = set() # 標籤集

    def load_data(self, file):
        for line in open(file):
            fields = line.strip().split()
            if len(fields) <2: continue # 特徵數要大於兩列
            label = fields[0] # 默認第一列式是標籤
            self.labels.add(label)
            for f in set(fields[1:]):
                self.feats[(label,f)] += 1 # (label,f)元組是特徵
                print(label,f)
            self.trainset.append(fields)


    # 模型訓練
    def train(self, max_iter= 1000): # 訓練樣本的主函數(迭代次數默認為＝1000次)
        self._initparams() # 初始化參數
        for i in range(max_iter):
            print('iter %d ...' % (i+1))
            self.ep = self.Ep() # 計算模型分布的特徵期望
            self.lastw = self.w[:]
            for i, win in enumerate(self.w):
                delta = 1.0/self.M * math.log(self.ep_[i]/ self.ep[i])
                self.w[i] += delta # 更新 w
            print(self.w, self.feats)
            if self._convergence(self.lastw, self.w): # 判斷算法是否收斂
                break

    def _initparams(self): # 初始化參數
        self.size = len(self.trainset)
        self.M = max([len(record) -1 for record in self.trainset]) # GIS訓練算法 -> train()當中的M參數

        self.ep_ = [0,0]*len(self.feats)
        for i, f in enumerate(self.feats):
            self.ep_[i] = float(self.feats[f])/float(self.size) # 計算經驗分布的特徵期望
            self.feats[f] = i # 為每個特徵函數分配 id

        self.w = [0.0]*len(self.feats) # 初始化權重
        self.lastw = self.w



    def Ep(self): # 計算模型分布的特徵期望值
        ep = [0.0]*len(self.feats)
        for record in self.trainset: # 從訓練集中迭代輸出特徵
            features = record[1:]
            prob = self.calprob(features) # 計算條件機率 P(y|x)
            for f in features:
                for w,l in prob:
                    if (l,f) in self.feats: # 來自訓練數據的特徵
                        idx = self.feats[(l,f)] # 獲取特徵id
                        ep[idx] += w* (1.0/self.size) # sum(1/N * f(y,x)*p(y|x)), p(x) = 1/N
        return ep

    def _convergence(self,lastw,w): # 收斂-終止的條件
        for w1, w2 in zip(lastw,w):
            if abs(w1-w2) >= 0.01: return False
        return True


    # 模型預測
    def predict(self, input): # 預測函數
        features = input.strip().split()
        prob = self.calprob(features)
        prob.sort(reverse = True)
        return prob

    def probwgt(self, features, label): # 計算每個特徵權重的指數
        wgt = 0.0
        for f in features:
            if (label,f) in self.feats:
                wgt += self.w[self.feats[(label,f)]]
        return math.exp(wgt)

    def calprob(self, features): # 計算條件機率
        wgts = [(self.probwgt(features,l),l) for l in self.labels]
        Z = sum([w for w,l in wgts]) # 歸一化參數
        prob = [(w/Z,l) for w, l in wgts] # 機率向量
        return prob