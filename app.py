import gradio as gr
import cv2
import os
import tempfile
from ultralytics import YOLO
import time
import torch


def yolo_inference(image, video, model_type, model_id, image_size, conf_threshold, model_file):
    # model = yolo.from_pretrained(f'jameslahm/{model_id}')
    if model_type == 'DIY':
        # model = torch.load('model_params_with_structure.pth')
        model = YOLO(model_file)
        model.eval()
    elif model_type =='pretrained':
        model_name = model_id if ("v10" in model_id or "v13" in model_id ) else model_id.replace("v", "")
        model = YOLO("{}.pt".format(model_name))
    start = time.perf_counter()
    if image:
        results = model.predict(source=image, imgsz=image_size, conf=conf_threshold)
        annotated_image = results[0].plot()
        end = time.perf_counter()
        print("Inference running time: {:.3f} ms".format((end - start) * 1000))
        return annotated_image[:, :, ::-1], None
    else:
        video_path = tempfile.mktemp(suffix=".webm")
        with open(video_path, "wb") as f:
            with open(video, "rb") as g:
                f.write(g.read())

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_video_path = tempfile.mktemp(suffix=".webm")
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'vp80'), fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame, imgsz=image_size, conf=conf_threshold)
            # results = model("")
            annotated_frame = results[0].plot()
            out.write(annotated_frame)

        cap.release()
        out.release()

        end = time.perf_counter()
        print("Inference running time: {:.3f} ms".format((end - start) * 1000))
        return None, output_video_path


def app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Image", visible=True)
                video = gr.Video(label="Video", visible=False)
                input_type = gr.Radio(
                    choices=["Image", "Video"],
                    value="Image",
                    label="Input Type",
                )
                model_type = gr.Radio(
                    choices=["pretrained", "DIY"],
                    value="pretrained",
                    label="Model Type",
                )
                model_file = gr.File(label="训练好的模型架构及参数文件", visible=False)

                model_id = gr.Dropdown(
                    label="Pretrained Model",
                    choices=[
                        "yolov10n",
                        "yolov10s",
                        "yolov10m",
                        "yolov10b",
                        "yolov10l",
                        "yolov10x",
                        "yolov11n",
                        "yolov11s",
                        "yolov11m",
                        "yolov11b",
                        "yolov11l",
                        "yolov11x",
                        "yolov12n",
                        "yolov12s",
                        "yolov12m",
                        "yolov12b",
                        "yolov12l",
                        "yolov12x",
                        "yolov13n",
                        "yolov13s",
                        "yolov13l",
                        "yolov13x",
                    ],
                    value="yolov12n",
                )
                image_size = gr.Slider(
                    label="Image Size",
                    minimum=320,
                    maximum=1280,
                    step=32,
                    value=640,
                )
                conf_threshold = gr.Slider(
                    label="Confidence Threshold",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.25,
                )
                yolo_infer = gr.Button(value="Detect Objects")

            with gr.Column():
                output_image = gr.Image(type="numpy", label="Annotated Image", visible=True)
                output_video = gr.Video(label="Annotated Video", visible=False)
                frame_interval = gr.Slider(
                    label="frame_interval 每隔多少帧保存一次图片，默认为1（即保存每一帧）",
                    minimum=1,
                    maximum=240,
                    step=1,
                    value=1,
                    visible=False
                )
                output_folder = gr.Textbox(label="保存图片的文件夹路径", placeholder="请输入文件夹路径", visible=False)
                video_extract = gr.Button(value="Video to Pictures", visible=False)
                extract_result_text = gr.Textbox(label="视频转换图片集操作结果", placeholder="操作结果将显示在这里", visible=False)

        def update_visibility(input_type):
            image = gr.update(visible=True) if input_type == "Image" else gr.update(visible=False)
            video = gr.update(visible=False) if input_type == "Image" else gr.update(visible=True)
            video_extract = gr.update(visible=False) if input_type == "Image" else gr.update(visible=True)
            output_folder = gr.update(visible=False) if input_type == "Image" else gr.update(visible=True)
            extract_result_text = gr.update(visible=False) if input_type == "Image" else gr.update(visible=True)
            frame_interval = gr.update(visible=False) if input_type == "Image" else gr.update(visible=True)
            output_image = gr.update(visible=True) if input_type == "Image" else gr.update(visible=False)
            output_video = gr.update(visible=False) if input_type == "Image" else gr.update(visible=True)

            return image, video, video_extract, output_folder, extract_result_text, frame_interval, output_image, output_video

        input_type.change(
            fn=update_visibility,
            inputs=[input_type],
            outputs=[image, video, video_extract, output_folder, extract_result_text, frame_interval, output_image, output_video],
        )

        def select_diy_model_and_params(model_type):
            model_file = gr.update(visible=True) if model_type == "DIY" else gr.update(visible=False)
            model_id = gr.update(visible=False) if model_type == "DIY" else gr.update(visible=True)
            return model_file, model_id

        model_type.change(
            fn=select_diy_model_and_params,
            inputs=[model_type],
            outputs=[model_file, model_id],
        )

        def run_inference(image, video, model_type, model_id, image_size, conf_threshold, input_type, model_file=None):
            if input_type == "Image":
                return yolo_inference(image, None, model_type, model_id, image_size, conf_threshold, model_file)
            else:
                return yolo_inference(None, video, model_type, model_id, image_size, conf_threshold, model_file)


        yolo_infer.click(
            fn=run_inference,
            inputs=[image, video, model_type, model_id, image_size, conf_threshold, input_type, model_file],
            outputs=[output_image, output_video],
        )

        def extract_video(video, output_folder, frame_interval):
            """
                从视频中提取帧并保存为图片数据集

                参数:
                    video (str): 视频文件
                    output_folder (str): 保存图片的文件夹路径
                    frame_interval (int): 每隔多少帧保存一次图片，默认为1（即保存每一帧）
                """
            # 创建输出文件夹
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # 打开视频文件
            cap = cv2.VideoCapture(video)
            if not cap.isOpened():
                print("无法打开视频文件")
                return

            frame_count = 0  # 当前帧计数器
            saved_count = 0  # 已保存的图片计数器

            while cap.isOpened():
                ret, frame = cap.read()  # 读取一帧
                if not ret:
                    break  # 如果读取失败，退出循环

                # 每隔 frame_interval 帧保存一次
                if frame_count % frame_interval == 0:
                    # 构造图片文件名
                    filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
                    cv2.imwrite(filename, frame)  # 保存图片
                    saved_count += 1
                    print(f"保存图片: {filename}")

                frame_count += 1

            cap.release()  # 释放视频文件
            print(f"总共保存了 {saved_count} 张图片到 {output_folder}")
            return "Extract success,\n totally saved {} pictures from video \n which has been stored  in {}".format(saved_count, output_folder)

        video_extract.click(
            fn=extract_video,
            inputs = [video, output_folder,frame_interval],
            outputs = [extract_result_text]
        )


gradio_app = gr.Blocks()
with gradio_app:
    gr.HTML(
        """
    <h1 style='text-align: center'>
    yolo: Real-Time End-to-End Object Detection
    </h1>
    """)
    with gr.Row():
        with gr.Column():
            app()
if __name__ == '__main__':
    gradio_app.launch()
