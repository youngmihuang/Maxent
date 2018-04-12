# coding: utf-8
from collections import defaultdict
import math
import copy

class MaxEnt(object):
    def __init__(self):
        self.feats = defaultdict(int)
        self.trainset = [] # 訓練集
        self.labels = set() # 標籤集
        self.size = int() # 訓練集大小
        self.M = int() # 訓練樣本中最大的特徵個數
        self.ep_ = float() # 計算經驗分布的特徵期望
        self.w = float() # 初始化權重
        self.ep = float() # 計算模型分布的特徵期望
        self.lastw = float() # 收斂時的權重
        self.feats_id = defaultdict(int) # 取出特徵位置的index


    def load_data(self, file):
        for line in open(file):
            fields = line.strip().split()
            if len(fields) <=3: continue # 每一筆要大於兩個特徵
            label = fields[0] 
            self.labels.add(label)
            for f in set(fields[1:]):
                self.feats[(label,f)] += 1 # (label,f)元組是特徵
                print(label,f)
            self.trainset.append(fields)

    # 模型訓練
    def train(self, max_iter= 1000): # 訓練樣本的主函數(迭代次數默認為＝1000次)
        self._initparams()
        for i in range(max_iter):
            print('iter %d ...' % (i+1))
            self.ep = self._expectedValue() # 計算模型分布的特徵期望
            self.lastw = self.w[:]
            for i, win in enumerate(self.w):
                delta = 1.0/self.M * math.log(self.ep_[i]/ self.ep[i])
                self.w[i] += delta # 更新 w
            print(self.w, self.feats)
            if self._convergence(self.lastw, self.w): # 判斷算法是否收斂
                break

    def _initparams(self): # 初始化參數
        self.size = len(self.trainset)
        self.M = max([len(record) -1 for record in self.trainset]) # 訓練樣本中最大的特徵個數
        self.ep_ = [0.0]*len(self.feats)

        for i, f in enumerate(self.feats):
            counts = self.feats[f]
            self.ep_[i] = float(counts)/float(self.size) # 計算經驗分布的特徵期望
            self.feats_id[f] = i # 為每個特徵函數分配id    
        self.w = [0.0]*len(self.feats) # 初始化權重
        
    def _expectedValue(self): # 特徵函數
        ep = [0.0]*len(self.feats)
        for record in self.trainset: # 從訓練集中迭代輸出特徵
            features = record[1:]
            prob = self._calprob(features) # 計算條件機率 P(y|x)
            for f in features:
                for w,label in prob:
                    if (label,f) in self.feats: # 來自訓練數據的特徵
                        idx = self.feats_id[(label,f)] # 獲取特徵id
                        ep[idx] += w* (1.0/self.size) 
        return ep

    def _calprob(self, features): # 計算條件機率
        wgts = [(self._probwgt(features,label),label) for label in self.labels]
        Z = sum([w for w,label in wgts]) 
        prob = [(w/Z,label) for w, label in wgts] 
        return prob

    def _probwgt(self, features, label): # 計算每個特徵權重的weight
        wgt = 0.0
        for f in features:
            if (label,f) in self.feats:
                wgt += self.w[self.feats_id[(label,f)]]
        return math.exp(wgt)

    def _convergence(self,lastw,w): # 收斂-終止的條件
        for w1, w2 in zip(lastw,w):
            if abs(w1-w2) >= 0.01: return False
        return True

    # 模型預測
    def predict(self, input): # 預測函數
        features = input.strip().split()
        prob = self._calprob(features)
        prob.sort(reverse = True)
        return prob

