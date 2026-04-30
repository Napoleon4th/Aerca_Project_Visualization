# Aerca_Project_Visualization

The productization visualization of the aerca project

原论文代码请参考：https://github.com/hanxiao0607/AERCA

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行
在项目文件夹下运行streamlit run app.py  

### 可视化代码app.py的调用逻辑
#### import部分
```python
from datasets import linear, lotka_volterra, lorenz96, swat, nonlinear, msds
from args import linear_args, lotka_volterra_args, lorenz96_args, swat_args, msds_args, nonlinear_args
from models import aerca
```

调用datasets/ 目录下的 6 个数据集生成文件 + args/ 目录下的参数文件 + models/aerca.py模型训练测试代码

