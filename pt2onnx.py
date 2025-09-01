from ultralytics import YOLO
import argparse
import ast
from pkl_util import save_dict_pickle




def parse_list_arg(list_arg):
    try:
        return ast.literal_eval(list_arg)
    except (ValueError, SyntaxError) as e:
        raise argparse.ArgumentTypeError(f"Invalid list argument: {list_arg}")


def convert(model_file, classes=None):
    # 加载YOLO模型
    model = YOLO(model_file)  # 替换为你的PyTorch模型路径

    if classes is not None:
        print('input classes: ', classes)
        #  修改模型的类别名称和数量
        # 假设我们只想保留 wzdkBRT 中的 1(gate_unnormal), 2(litter) 和 5(screen_unnormal) 类别
        desired_class_ids = classes if (len(classes)>0) else [1, 2, 5]  # 要保留的类别ID
        desired_class_names = {model.names[i] for i in desired_class_ids}  # 对应的类别名称

        desired_class_name_map = {i: model.names[i] for i in desired_class_ids}  # 对应的类别名称字典
        save_dict_pickle(desired_class_name_map, "class_name_map.pkl")

        # 关键步骤：修改模型本身的属性
        model.model.names = desired_class_names  # 更新类别名称字典
        model.model.nc = len(desired_class_ids)  # 更新类别数量

        print("修改完成！ONNX 模型现在只预测，注意在项目文件中包含了class_name_map.pkl文件用于重定向类别id，不可删除", desired_class_names)

    # 导出为ONNX格式
    model.export(format="onnx", simplify=True)  # 创建 'best.onnx'


# ---------- 命令行 ----------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert trained model file to ONNX format.')
    parser.add_argument('--model', required=True, help='预训练好的模型文件')
    parser.add_argument('--classes', type=parse_list_arg, required=False, help='限制模型输出的预测类别')
    args = parser.parse_args()
    convert(args.model, args.classes)
