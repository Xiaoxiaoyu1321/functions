import re
import numpy as np
import matplotlib.pyplot as plt


def read_functions_from_txt(file_path):
    """
    从文本文件中读取函数表达式和定义域
    :param file_path: 文本文件的路径
    :return: 函数表达式和定义域的列表
    """
    functions = []
    with open(file_path, 'r') as file:
        for line in file:
            # 使用正则表达式提取函数表达式和定义域
            match = re.search(r'y = (.*?) （(.*?)）', line)
            if match:
                func_str = match.group(1)
                domain_str = match.group(2)
                functions.append((func_str, domain_str))
    return functions


def plot_functions(functions):
    """
    绘制函数图像
    :param functions: 函数表达式和定义域的列表
    """
    plt.figure(figsize=(10, 8))

    for func_str, domain_str in functions:
        # 提取定义域的上下限
        lower_bound, upper_bound = map(float, re.findall(r'[-+]?\d*\.\d+|\d+', domain_str))
        x = np.linspace(lower_bound, upper_bound, 400)

        try:
            # 定义一个函数用于计算 y 值
            def func(x_val):
                return eval(func_str.replace('x', str(x_val)))

            y = np.vectorize(func)(x)
            plt.plot(x, y, label=f'{func_str} ({domain_str})')
        except Exception as e:
            print(f"绘制函数 {func_str} 时出错: {e}")

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('函数图像')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    file_path = 'output_for_sketchpad.txt'  # 替换为实际的 txt 文件路径
    functions = read_functions_from_txt(file_path)
    plot_functions(functions)