# Some synthetic datasets with linear dynamics
import numpy as np
import os
import random


class Linear:
    def __init__(self, options):
        self.options = options
        self.data_dict = {}
        self.seed = options['seed']
        self.n = options['training_size'] + options['testing_size']
        self.t = options['T']
        self.mul = options.get('mul', 4.5)
        self.a = options['a'] if options['a'] is not None else self._generate_random_coefficients()
        self.adlength = options.get('adlength', 60)
        self.adtype = options.get('adtype', 'spike')
        self.data_dir = options['data_dir']
        self.dependent_features = options.get('dependent_features', 0)

        self.supported_adtypes = ['spike', 'step', 'causal']

        if self.adtype not in self.supported_adtypes:
            print(f"Warning: adtype '{self.adtype}' not supported. Using 'spike' instead.")
            self.adtype = 'spike'

    def _generate_random_coefficients(self):
        a = np.zeros((8,))
        for k in range(8):
            u_1 = np.random.uniform(0, 1)
            a[k] = np.random.uniform(-0.8, -0.2) if u_1 <= 0.5 else np.random.uniform(0.2, 0.8)
        return a

    def generate_example(self):
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)

        x_n_list = np.zeros((self.n, self.t, 4))
        x_ab_list = np.zeros((self.n, self.t, 4))
        label_list = np.zeros((self.n, self.t, 4))

        for i in range(self.n):
            eps = 0.4 * np.random.randn(self.t, 4)

            # 正常序列
            x = np.zeros(self.t)
            w = np.zeros(self.t)
            y = np.zeros(self.t)
            z = np.zeros(self.t)
            for j in range(1, self.t):
                x[j] = self.a[0] * x[j - 1] + eps[j, 0]
                w[j] = self.a[1] * w[j - 1] + self.a[2] * x[j - 1] + eps[j, 1]
                y[j] = self.a[3] * y[j - 1] + self.a[4] * w[j - 1] + eps[j, 2]
                z[j] = self.a[5] * z[j - 1] + self.a[6] * w[j - 1] + self.a[7] * y[j - 1] + eps[j, 3]

            x_n_list[i] = np.stack((x, w, y, z), axis=-1)

            # ====================== 生成异常（最终稳定版） ======================
            start = np.random.randint(int(0.2 * self.t), int(0.8 * self.t - self.adlength))
            t_p = np.arange(start, start + self.adlength)
            end = start + 30
            # 安全选择受影响的变量（1~3个）
            num_features = np.random.randint(1, 4)
            feature_p = np.random.choice(4, size=num_features, replace=False)

            temp_label = np.zeros((self.t, 4))
            temp_label[np.ix_(t_p, feature_p)] = 1

            x_ab = x.copy()
            w_ab = w.copy()
            y_ab = y.copy()
            z_ab = z.copy()

            if self.adtype == 'spike':
                amp = self.mul * 2.0
                for f in feature_p:
                    if f == 0:
                        x_ab[t_p] += amp
                    elif f == 1:
                        w_ab[t_p] += amp
                    elif f == 2:
                        y_ab[t_p] += amp
                    elif f == 3:
                        z_ab[t_p] += amp

            elif self.adtype == 'step':
                # Step：从异常开始位置一直持续到序列结束
                step_value = self.mul * 2.0
                for f in feature_p:
                    if f == 0:
                        x_ab[start:end] += step_value
                    elif f == 1:
                        w_ab[start:end] += step_value
                    elif f == 2:
                        y_ab[start:end] += step_value
                    elif f == 3:
                        z_ab[start:end] += step_value



            elif self.adtype == 'causal':
                b = self.a * 4.8
                for j in range(1, self.t):
                    if start <= j < start + self.adlength:
                        x_ab[j] = b[0] * x_ab[j - 1] + eps[j, 0]
                        w_ab[j] = b[1] * w_ab[j - 1] + b[2] * x_ab[j - 1] + eps[j, 1]
                        y_ab[j] = b[3] * y_ab[j - 1] + b[4] * w_ab[j - 1] + eps[j, 2]
                        z_ab[j] = b[5] * z_ab[j - 1] + b[6] * w_ab[j - 1] + b[7] * y_ab[j - 1] + eps[j, 3]
                    else:
                        x_ab[j] = self.a[0] * x_ab[j - 1] + eps[j, 0]
                        w_ab[j] = self.a[1] * w_ab[j - 1] + self.a[2] * x_ab[j - 1] + eps[j, 1]
                        y_ab[j] = self.a[3] * y_ab[j - 1] + self.a[4] * w_ab[j - 1] + eps[j, 2]
                        z_ab[j] = self.a[5] * z_ab[j - 1] + self.a[6] * w_ab[j - 1] + self.a[7] * y_ab[j - 1] + eps[
                            j, 3]

            x_ab_list[i] = np.stack((x_ab, w_ab, y_ab, z_ab), axis=-1)
            label_list[i] = temp_label

        self.data_dict = {
            'x_n_list': x_n_list,
            'x_ab_list': x_ab_list,
            'label_list': label_list,
            'a': self.a,
            'causal_struct': np.array([[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0], [0, 1, 1, 1]]),
            'causal_struct_value': np.array(
                [[self.a[0], 0, 0, 0], [self.a[2], self.a[1], 0, 0], [0, self.a[4], self.a[3], 0],
                 [0, self.a[6], self.a[7], self.a[5]]]),
            'signed_causal_struct': np.sign(np.array(
                [[self.a[0], 0, 0, 0], [self.a[2], self.a[1], 0, 0], [0, self.a[4], self.a[3], 0],
                 [0, self.a[6], self.a[7], self.a[5]]]))
        }

    def save_data(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        np.save(os.path.join(self.data_dir, 'x_n_list.npy'), self.data_dict['x_n_list'])
        np.save(os.path.join(self.data_dir, 'x_ab_list.npy'), self.data_dict['x_ab_list'])
        np.save(os.path.join(self.data_dir, 'label_list.npy'), self.data_dict['label_list'])
        np.save(os.path.join(self.data_dir, 'a.npy'), self.data_dict['a'])
        np.save(os.path.join(self.data_dir, 'causal_struct.npy'), self.data_dict['causal_struct'])
        np.save(os.path.join(self.data_dir, 'causal_struct_value.npy'), self.data_dict['causal_struct_value'])
        np.save(os.path.join(self.data_dir, 'signed_causal_struct.npy'), self.data_dict['signed_causal_struct'])

    def load_data(self):
        self.data_dict['x_n_list'] = np.load(os.path.join(self.data_dir, 'x_n_list.npy'))
        self.data_dict['x_ab_list'] = np.load(os.path.join(self.data_dir, 'x_ab_list.npy'))
        self.data_dict['label_list'] = np.load(os.path.join(self.data_dir, 'label_list.npy'))
        self.data_dict['a'] = np.load(os.path.join(self.data_dir, 'a.npy'))
        self.data_dict['causal_struct'] = np.load(os.path.join(self.data_dir, 'causal_struct.npy'))
        self.data_dict['causal_struct_value'] = np.load(os.path.join(self.data_dir, 'causal_struct_value.npy'))
        self.data_dict['signed_causal_struct'] = np.load(os.path.join(self.data_dir, 'signed_causal_struct.npy'))