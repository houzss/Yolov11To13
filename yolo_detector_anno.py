#!/usr/bin/env python3
"""
基于 Ultralytics YOLOv8 的目标检测与 PASCAL VOC 标签生成工具
使用训练好的 best.pt 模型进行推理，并生成 VOC 格式的 XML 标注文件。

1. 单张图片检测
bash
python yolo_voc_detector.py --model best.pt --source path/to/your/image.jpg --output output_dir --save-img
2. 批量图片检测
bash
python yolo_voc_detector.py --model best.pt --source path/to/image/folder --output output_dir
3. 调整检测参数
bash
# 提高置信度阈值以减少误检
python yolo_voc_detector.py --model best.pt --source image.jpg --conf 0.5

# 调整IOU阈值
python yolo_voc_detector.py --model best.pt --source image.jpg --iou 0.6

参数说明
--model / -m: 训练好的 YOLOv8 模型路径 (best.pt)

--source / -s: 输入源（单张图片或图片目录）

--output / -o: XML 输出目录（可选）

--conf / -c: 置信度阈值（默认: 0.25）

--iou / -i: IOU 阈值（默认: 0.45）

--save-img: 是否保存带检测框的可视化图片

生成的 VOC XML 格式
生成的 XML 文件包含：

图片基本信息（路径、尺寸）

每个检测对象的边界框坐标

类别名称和置信度

符合 PASCAL VOC 标准的格式

"""

import argparse
import cv2
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from ultralytics import YOLO
import numpy as np


class YOLOVOCDetector:
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45):
        """
        初始化 YOLOv8 检测器

        参数:
            model_path: 训练好的模型路径 (.pt)
            conf_threshold: 置信度阈值
            iou_threshold: IOU 阈值
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # 加载训练好的 YOLOv8 模型
        self.model = YOLO(model_path)
        print(f"[信息] 成功加载 YOLOv8 模型: {model_path}")

        # 获取类别名称
        self.classes = self.model.names
        print(f"[信息] 模型包含 {len(self.classes)} 个类别: {list(self.classes.values())}")

    def detect_image(self, image_path, output_dir=None, save_image=False):
        """
        检测单张图片并生成 VOC 标签

        参数:
            image_path: 输入图片路径
            output_dir: XML 输出目录 (None 则与图片同目录)
            save_image: 是否保存带检测框的可视化图片

        返回:
            results: Ultralytics 检测结果
            xml_path: 生成的 XML 文件路径
        """
        # 检查图片是否存在
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件不存在: {image_path}")

        # 使用模型进行预测
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            save=save_image,  # 是否保存检测结果图片
            save_txt=False,  # 不保存YOLO格式的标签
            save_conf=True,  # 保存置信度
            show=False  # 不显示图片
        )

        # 处理检测结果
        if results and len(results) > 0:
            result = results[0]  # 第一张图片的结果
            xml_path = self._generate_voc_xml(image_path, result, output_dir)
            return result, xml_path
        else:
            print(f"[警告] 未检测到任何目标: {image_path}")
            # 即使没有检测到目标，也生成空的XML文件
            xml_path = self._generate_voc_xml(image_path, None, output_dir)
            return None, xml_path

    def detect_directory(self, image_dir, output_dir=None, save_image=False):
        """
        批量检测目录中的所有图片

        参数:
            image_dir: 图片目录路径
            output_dir: XML 输出目录
            save_image: 是否保存可视化图片
        """
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"图片目录不存在: {image_dir}")

        # 支持的图片格式
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        image_paths = []

        for ext in image_extensions:
            image_paths.extend(list(image_dir.glob(f'*{ext}')))
            image_paths.extend(list(image_dir.glob(f'*{ext.upper()}')))

        print(f"[信息] 找到 {len(image_paths)} 张图片")

        for img_path in image_paths:
            try:
                result, xml_path = self.detect_image(
                    str(img_path), output_dir, save_image
                )
                if result and len(result.boxes) > 0:
                    print(f"[成功] 检测到 {len(result.boxes)} 个目标: {img_path.name} -> {xml_path}")
                else:
                    print(f"[信息] 未检测到目标: {img_path.name}")
            except Exception as e:
                print(f"[错误] 处理图片 {img_path.name} 时出错: {e}")

    def _generate_voc_xml(self, image_path, result, output_dir=None):
        """
        生成 PASCAL VOC 格式的 XML 文件
        """
        # 确定输出路径
        if output_dir is None:
            output_dir = os.path.dirname(image_path)
        else:
            os.makedirs(output_dir, exist_ok=True)

        image_name = os.path.basename(image_path)
        xml_filename = os.path.splitext(image_name)[0] + '.xml'
        xml_path = os.path.join(output_dir, xml_filename)

        # 读取图片获取尺寸
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")
        height, width, depth = img.shape

        # 创建 XML 结构
        annotation = ET.Element('annotation')

        # 添加文件夹和文件名
        folder = ET.SubElement(annotation, 'folder')
        folder.text = os.path.basename(os.path.dirname(image_path))

        filename = ET.SubElement(annotation, 'filename')
        filename.text = image_name

        # 添加路径
        path = ET.SubElement(annotation, 'path')
        path.text = os.path.abspath(image_path)

        # 添加源信息
        source = ET.SubElement(annotation, 'source')
        database = ET.SubElement(source, 'database')
        database.text = 'Unknown'

        # 添加图片尺寸
        size = ET.SubElement(annotation, 'size')
        width_elem = ET.SubElement(size, 'width')
        width_elem.text = str(width)
        height_elem = ET.SubElement(size, 'height')
        height_elem.text = str(height)
        depth_elem = ET.SubElement(size, 'depth')
        depth_elem.text = str(depth)

        # 添加分割标记
        segmented = ET.SubElement(annotation, 'segmented')
        segmented.text = '0'

        # 如果有检测结果，添加检测对象
        if result and hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes
            for i in range(len(boxes)):
                box = boxes[i]
                class_id = int(box.cls.item())
                confidence = box.conf.item()
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # 转换为整数坐标
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # 创建对象节点
                obj = ET.SubElement(annotation, 'object')

                # 类别名称
                name_elem = ET.SubElement(obj, 'name')
                name_elem.text = self.classes[class_id]

                # 姿态（通常为Unspecified）
                pose_elem = ET.SubElement(obj, 'pose')
                pose_elem.text = 'Unspecified'

                # 截断（通常为0）
                truncated_elem = ET.SubElement(obj, 'truncated')
                truncated_elem.text = '0'

                # 难例（通常为0）
                difficult_elem = ET.SubElement(obj, 'difficult')
                difficult_elem.text = '0'

                # 置信度（自定义字段，非VOC标准）
                confidence_elem = ET.SubElement(obj, 'confidence')
                confidence_elem.text = f'{confidence:.4f}'

                # 边界框
                bndbox_elem = ET.SubElement(obj, 'bndbox')
                xmin_elem = ET.SubElement(bndbox_elem, 'xmin')
                xmin_elem.text = str(x1)
                ymin_elem = ET.SubElement(bndbox_elem, 'ymin')
                ymin_elem.text = str(y1)
                xmax_elem = ET.SubElement(bndbox_elem, 'xmax')
                xmax_elem.text = str(x2)
                ymax_elem = ET.SubElement(bndbox_elem, 'ymax')
                ymax_elem.text = str(y2)

        # 美化XML格式并写入文件
        self._prettify_and_save_xml(annotation, xml_path)
        return xml_path

    def _prettify_and_save_xml(self, elem, xml_path):
        """美化XML格式并保存到文件"""
        rough_string = ET.tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        with open(xml_path, 'w', encoding='utf-8') as f:
            reparsed.writexml(f, encoding='utf-8', indent='  ')


def main():
    parser = argparse.ArgumentParser(description='基于 YOLOv8 的目标检测与 VOC 标签生成工具')
    parser.add_argument('--model', '-m', required=True, help='训练好的 YOLOv8 模型路径 (.pt 文件)')
    parser.add_argument('--source', '-s', required=True, help='输入源：单张图片路径或图片目录路径')
    parser.add_argument('--output', '-o', default=None, help='XML 输出目录 (默认与图片同目录)')
    parser.add_argument('--conf', '-c', type=float, default=0.25, help='置信度阈值 (默认: 0.25)')
    parser.add_argument('--iou', '-i', type=float, default=0.45, help='IOU 阈值 (默认: 0.45)')
    parser.add_argument('--save-img', action='store_true', help='保存带检测框的可视化图片')

    args = parser.parse_args()

    # 初始化检测器
    detector = YOLOVOCDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )

    # 判断输入源类型
    source_path = Path(args.source)
    if source_path.is_file():
        # 单张图片检测
        print(f"[信息] 开始处理单张图片: {args.source}")
        result, xml_path = detector.detect_image(
            args.source, args.output, args.save_img
        )
        if result and len(result.boxes) > 0:
            print(f"[成功] 检测完成! 生成 VOC 标签: {xml_path}")
        else:
            print(f"[信息] 检测完成! 未检测到目标，生成空标签: {xml_path}")

    elif source_path.is_dir():
        # 批量图片检测
        print(f"[信息] 开始批量处理目录: {args.source}")
        detector.detect_directory(
            args.source, args.output, args.save_img
        )
        print("[信息] 批量处理完成!")
    else:
        print(f"[错误] 输入路径不存在: {args.source}")


if __name__ == "__main__":
    # 由于 prettify 需要 minidom，在这里导入
    from xml.dom import minidom
    main()