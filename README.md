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

#### 数据集准备与可视化

在第110行开始，运行run_button后进行数据准备：
```python
DataClass = dataset_mapping[dataset_name]["class"]
data_class = DataClass(options)          # 实例化数据集类

if preprocessing_data == 1:
    data_class.generate_example()        # 数据集类的函数 generate_example
    data_class.save_data()               # 数据集类的函数 save_data
else:
    data_class.load_data()               # 如果加载已有数据，调用数据集类的函数 load_data，这里还没完全写好，演示时候都用的是直接生成数据再训练测试的
```

调用各个 datasets/xxx.py 文件中的类方法

#### 模型完整运行阶段

在第318行
```python
from main import main as run_main          # 动态导入 main.py
results = run_main(sys.argv)               # 调用 main.py 的 main() 函数
```

通过 sys.argv hack，把参数伪装成命令行参数传给 main.py

**main() 函数内部流程：**

创建 AERCA 模型对象

调用 aerca_model._training(...)

调用 aerca_model._testing_causal_discover(...)

调用 aerca_model._testing_root_cause(...)

返回结构化 results 字典内容

**返回results包括：**

'root_cause_results' ← 来自 aerca.py 的 _testing_root_cause

'causal_results' ← 来自 aerca.py 的 _testing_causal_discover

'test_x_ab', 'test_label', 'test_causal' 等用于可视化的内容

#### 可视化部分使用的数据来源总结

| 可视化区块          | 使用的数据来源                                      | 具体来自哪里                          |
|---------------------|----------------------------------------------------|---------------------------------------|
| **正常 vs 异常对比图** | `data_class.data_dict['x_n_list']` 和 `test_x_ab` | 数据集类 + `main.py` 返回的 `results` |
| **正常/异常时间序列**   | `data_class.data_dict['x_n_list']` / `test_x_ab`  | 同上                                  |
| **异常位置热力图**      | `data_class.data_dict['label_list']` 或 `test_label` | 同上                                  |
| **真实因果矩阵**        | `data_class.data_dict['causal_struct']`            | 数据集类                              |
| **根因高亮图**          | `test_x_ab` + `results['predicted_root_causes']`   | `main.py` 返回的 `root_cause_results` |
| **因果发现矩阵对比**    | `results['true_causal_matrix']` 和 `predicted_causal_matrix` | `main.py` 返回的 `causal_results`    |

