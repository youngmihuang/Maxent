# coding: utf-8
# import maxent
import maxent
model = maxent.MaxEnt()
model.load_data('data.txt') # 導入數據集
model.train() # 訓練模型
# 可替換案例
print(model.predict('Rainy Happy Dry'))
