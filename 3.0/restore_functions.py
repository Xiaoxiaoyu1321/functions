import numpy as np
import matplotlib.pyplot as plt
import pickle


def linear_func(x, a, b):
    return a * x + b


def restore_functions(input_pickle='functions.pkl'):
    with open(input_pickle, 'rb') as f:
        functions = pickle.load(f)

    #
    plt.figure(figsize=(8, 8))
    for params, (x_start, x_end) in functions:
        x_vals = np.linspace(x_start, x_end, 100)
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