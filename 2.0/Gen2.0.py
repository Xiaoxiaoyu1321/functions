import numpy as np
import matplotlib.pyplot as plt
import pickle


def linear_func(x, a, b):
    """ 定义线性函数用于拟合 """
    return a * x + b


def restore_functions(input_pickle='functions.pkl'):
    # 使用 pickle 读取保存的函数数据
    with open(input_pickle, 'rb') as f:
        functions = pickle.load(f)

    # 绘制还原后的函数图像
    plt.figure(figsize=(8, 8))
    for params, (x_start, x_end) in functions:
        # 增加点数，让图像更细致
        x_vals = np.linspace(x_start, x_end, 1000)
        y_vals = linear_func(x_vals, *params)
        plt.plot(x_vals, y_vals, 'r-', linewidth=0.5)

    plt.title('Restored Functions')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    return functions


if __name__ == "__main__":
    restored_functions = restore_functions()
    print("还原后的函数参数：", restored_functions)  