import os
import argparse


def compare_and_delete(folder_a, folder_b):
    """
    比较两个文件夹，从B文件夹中删除A文件夹中不存在的对应文件

    参数:
        folder_a (str): 存放图片的文件夹路径
        folder_b (str): 存放文本文件的文件夹路径
    """
    # 支持的图片扩展名
    image_extensions = {'.jpg', '.jpeg', '.png'}

    # 获取文件夹A中的所有图片文件的基本名（不带扩展名）
    image_basenames = set()
    for filename in os.listdir(folder_a):
        basename, ext = os.path.splitext(filename)
        if ext.lower() in image_extensions:
            image_basenames.add(basename)

    # 遍历文件夹B，删除没有对应图片的文本文件
    deleted_count = 0
    for filename in os.listdir(folder_b):
        basename, ext = os.path.splitext(filename)
        if ext.lower() == '.txt' and basename not in image_basenames:
            file_path = os.path.join(folder_b, filename)
            os.remove(file_path)
            print(f"已删除: {filename}")
            deleted_count += 1

    print(f"操作完成。共删除了 {deleted_count} 个文件。")


def main():
    parser = argparse.ArgumentParser(description='比较图片文件夹和文本文件夹，删除没有对应图片的文本文件')
    parser.add_argument('--picPath', help='存放图片的文件夹路径')
    parser.add_argument('--txtPath', help='存放文本文件的文件夹路径')

    args = parser.parse_args()

    # 检查文件夹是否存在
    if not os.path.isdir(args.picPath):
        print(f"错误: 文件夹A '{args.folder_a}' 不存在")
        return

    if not os.path.isdir(args.txtPath):
        print(f"错误: 文件夹B '{args.folder_b}' 不存在")
        return

    # 执行比较和删除操作
    compare_and_delete(args.picPath, args.txtPath)


if __name__ == "__main__":
    main()