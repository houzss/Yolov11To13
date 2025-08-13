from ultralytics import YOLO
import argparse

def convert(model_file):
    # 加载YOLO模型
    model = YOLO(model_file)  # 替换为你的PyTorch模型路径

    # 导出为ONNX格式
    model.export(format="onnx", simplify=True)  # 创建 'best.onnx'

# ---------- 命令行 ----------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert trained model file to ONNX format.')
    parser.add_argument('--model', required=True, help='预训练好的模型文件')
    args = parser.parse_args()
    convert(args.model)
