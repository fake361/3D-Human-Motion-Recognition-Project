

import glob
import os

# Windows兼容性：使用相对路径替代硬编码的Linux路径
weights_path = glob.glob(os.path.join('results', 'ntu_NTU60_CS', 'runs-110*'))[0]
print(1223)