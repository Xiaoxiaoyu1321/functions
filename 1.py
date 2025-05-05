import cv2
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from PIL import Image


def linear_func(x, a, b):
    """ 定义线性函数用于拟合 """
    return a * x + b


def image_to_functions(image_path, block_size=10, output_txt='output_for_sketchpad.txt'):
    # 读取图片
    image = Image.open(image_path).convert('L')
    image = np.array(image)

    # 边缘检测
    edges = cv2.Canny(image, 50, 150)

    height, width = image.shape
    functions = []

    # 分块处理
    for y_start in range(0, height, block_size):
        for x_start in range(0, width, block_size):
            y_end = min(y_start + block_size, height)
            x_end = min(x_start + block_size, width)
            block = edges[y_start:y_end, x_start:x_end]

            # 找到当前块中的边缘像素点
            y_pixels, x_pixels = np.where(block > 0)
            if len(x_pixels) > 2:
                x_pixels += x_start
                y_pixels += y_start
                try:
                    # 进行线性函数拟合
                    popt, _ = curve_fit(linear_func, x_pixels, y_pixels)
                    functions.append((popt, (x_start, x_end)))
                except RuntimeError:
                    continue

    # 绘制函数图像
    plt.figure(figsize=(8, 8))
    x_vals_all = np.linspace(0, width, 1000)
    for params, (x_start, x_end) in functions:
        x_vals = np.linspace(x_start, x_end, 100)
        y_vals = linear_func(x_vals, *params)
        plt.plot(x_vals, y_vals, 'r-', linewidth=0.5)

    plt.imshow(image, cmap='gray')
    plt.show()

    # 打开文本文件用于写入
    with open(output_txt, 'w') as txt_file:
        for i, (params, (x_start, x_end)) in enumerate(functions):
            a, b = params
            func_str = f"{a:.4f}*x + {b:.4f}"
            domain_str = f"x >= {x_start} && x < {x_end}"
            # 将函数表达式写入文本文件
            txt_str = f"函数 {i + 1}: y = {func_str} （{domain_str}）"
            txt_file.write(txt_str + '\n')

    print(f"适合几何画板使用的函数表达式已保存到 {output_txt}")

    return functions


if __name__ == "__main__":
    image_path = '2.jpg'  # 替换为你的图片路径
    functions = image_to_functions(image_path)
    print("拟合得到的函数参数：", functions)