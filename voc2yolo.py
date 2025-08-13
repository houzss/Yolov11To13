#!/usr/bin/env python3
"""
voc2yolo.py
把 PascalVOC XML 标注批量转换为 YOLO txt 格式。

Usage:
    python voc2yolo.py \
        --voc-root datasets/VOC2007/Annotations \  VOC格式的标注文件夹
        --yolo-root datasets/VOC2007/labels  目标生成yolo格式数据的文件夹
"""
import os
import argparse
from pathlib import Path
from lxml import etree


# ---------- 工具函数 ----------
def parse_xml(xml_path):
    """解析单个 VOC XML，返回 dict"""
    tree = etree.parse(str(xml_path))
    root = tree.getroot()

    size = root.find('size')
    width = int(size.findtext('width'))
    height = int(size.findtext('height'))

    boxes = []
    for obj in root.findall('object'):
        name = obj.findtext('name').strip()
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.findtext('xmin'))
        ymin = float(bndbox.findtext('ymin'))
        xmax = float(bndbox.findtext('xmax'))
        ymax = float(bndbox.findtext('ymax'))

        # 归一化到 0~1
        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height
        box_w = (xmax - xmin) / width
        box_h = (ymax - ymin) / height
        boxes.append((name, x_center, y_center, box_w, box_h))
    return boxes


def build_class_map(voc_root):
    """统计所有类别，生成 {name: id} 映射"""
    class_set = set()
    for xml_file in Path(voc_root).rglob('*.xml'):
        try:
            boxes = parse_xml(xml_file)
            for name, *_ in boxes:
                class_set.add(name)
        except Exception as e:
            print(f'[WARN] 解析 {xml_file} 失败：{e}')
    class_map = {cls: idx for idx, cls in enumerate(sorted(class_set))}
    return class_map


# ---------- 主逻辑 ----------
def convert(voc_root, yolo_root):
    voc_root = Path(voc_root)
    yolo_root = Path(yolo_root)
    yolo_root.mkdir(parents=True, exist_ok=True)

    class_map = build_class_map(voc_root)
    if not class_map:
        print('未检测到任何 XML 或目标，终止。')
        return

    # 写出类别文件
    classes_file = yolo_root / 'classes.txt'
    with open(classes_file, 'w', encoding='utf-8') as f:
        for cls in sorted(class_map, key=lambda x: class_map[x]):
            f.write(cls + '\n')
    print(f'类别映射已写入 {classes_file}：{class_map}')

    # 遍历 XML 并转换
    for xml_file in voc_root.rglob('*.xml'):
        try:
            boxes = parse_xml(xml_file)
        except Exception as e:
            print(f'跳过 {xml_file}：{e}')
            continue
        if not boxes:
            print(f'跳过 {xml_file}：无目标')
            continue

        # 构造输出 txt 路径：保持相对目录结构
        rel_path = xml_file.relative_to(voc_root).with_suffix('.txt')
        txt_path = yolo_root / rel_path
        txt_path.parent.mkdir(parents=True, exist_ok=True)

        with open(txt_path, 'w', encoding='utf-8') as f:
            for name, xc, yc, w, h in boxes:
                if name not in class_map:
                    print(f'[WARN] 类别 {name} 不在类别表中，跳过')
                    continue
                cls_id = class_map[name]
                f.write(f'{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n')
    print('转换完成！')


# ---------- 命令行 ----------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert PascalVOC XML to YOLO txt')
    parser.add_argument('--voc-root', required=True, help='PascalVOC Annotations 目录')
    parser.add_argument('--yolo-root', required=True, help='输出 YOLO txt 目录')
    args = parser.parse_args()
    convert(args.voc_root, args.yolo_root)