import os
import xml.etree.ElementTree as ET
from collections import defaultdict
import argparse


def convert_labels(input_dir, output_dir=None):
    """
    批量转换XML标签文件中的标签名称

    参数:
        input_dir: 输入XML文件目录
        output_dir: 输出XML文件目录(如果为None则覆盖原文件)
    """
    # 标签映射规则
    label_mapping = {
        "水污染": "oil_pollution",
        "废弃船": "ship_abandoned",
        "漂浮物": "floating_debris",
        "捕鱼养殖": "fishing",
        "废弃物": "riverbank_garbage"
    }

    # 收集未识别的标签
    unknown_labels = defaultdict(int)

    # 确保输出目录存在
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历输入目录中的所有XML文件
    xml_files = [f for f in os.listdir(input_dir) if f.endswith('.xml')]

    if not xml_files:
        print(f"在目录 {input_dir} 中没有找到XML文件")
        return

    print(f"找到 {len(xml_files)} 个XML文件")

    for xml_file in xml_files:
        input_path = os.path.join(input_dir, xml_file)

        try:
            # 解析XML文件
            tree = ET.parse(input_path)
            root = tree.getroot()

            # 查找所有对象标签
            changed = False
            for obj in root.findall('object'):
                name_elem = obj.find('name')
                if name_elem is not None:
                    original_name = name_elem.text

                    # 检查是否需要转换
                    if original_name in label_mapping:
                        name_elem.text = label_mapping[original_name]
                        changed = True
                        print(f"文件 {xml_file}: 将 '{original_name}' 转换为 '{label_mapping[original_name]}'")
                    elif original_name not in label_mapping.values():
                        # 记录未知标签
                        unknown_labels[original_name] += 1
                        print(f"文件'{xml_file}' 存在未识别的标签:'{original_name}' ")

            # 保存文件
            if output_dir:
                output_path = os.path.join(output_dir, xml_file)
                tree.write(output_path, encoding='utf-8', xml_declaration=True)
            elif changed:
                # 直接覆盖原文件
                tree.write(input_path, encoding='utf-8', xml_declaration=True)

        except Exception as e:
            print(f"处理文件 {xml_file} 时出错: {str(e)}")

    # 输出未识别的标签
    if unknown_labels:
        print("\n未识别的标签:")
        for label, count in unknown_labels.items():
            print(f"  {label}: 出现 {count} 次")
    else:
        print("\n所有标签都已识别并处理")


def main():
    parser = argparse.ArgumentParser(description='YOLO VOC标签批量转换工具')
    parser.add_argument('input_dir', help='输入XML文件目录')
    parser.add_argument('-o', '--output_dir', help='输出XML文件目录(可选，不指定则覆盖原文件)')

    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"错误: 目录 '{args.input_dir}' 不存在")
        return

    convert_labels(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()