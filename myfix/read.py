import pickle
import os
# Windows兼容性：使用相对路径替代硬编码的Linux路径
# filepath='data/NW-UCLA/val_label.pkl'
filepath = os.path.join('data', 'jtbar', 'label.pkl')
with open(filepath, 'rb') as f:
    data = pickle.load(f)
    numlen=len(data)
    print(12)