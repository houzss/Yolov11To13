# Yolov11To13
基于Ultralytics框架的Yolo目标检测（可集成v1~v13所有版本）
## 目标环境搭建
1. 安装cuda C:\Users\houzs\AppData\Local\Temp\cuda
2. 安装anaconda D:\anaconda3
anaconda虚拟环境位置 D:\anaconda3\envs
配置conda系统变量 D:\anaconda3 D:\anaconda3\Library\bin
D:\anaconda3\Scripts
3. 更换清华源
conda config
conda config --show channels
conda config --add channels
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels
https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
conda config --add channels
https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
4. 设置搜索时显⽰通道地址
conda config --set show_channel_urls yes
5. 配置pytorch环境 conda create -n pytorch_env python pytorch torchvision
conda activate pytorch_env
python
import torch
torch.cuda.is_available()
6. 配置yolov10环境
按照项⽬包安装依赖
conda create -n yolov10 python=3.9
conda activate yolov10
pip install -r requirements.txt
pip install -e .
pytorch是cpu版的，重新安装gpu版本
conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda -c pytorch -c nvidia

7. 配置yolov13环境
0、创建虚拟环境
conda create -n yolov13 python=3.11
conda activate yolov13
pip install -r requirements.txt
pip install -e .
1、按照项⽬包安装依赖
找⽹上⼤神的2.7.4的，但是torch要升到2.5.1 , 下载地址：
https://github.com/kingbri1/flash-attention/releases
conda install pytorch==2.5.1 torchvision==0.20.1 pytorch-cuda=12.4 -c
pytorch -c nvidia
下载whl轮⼦⽂件后放到⽬录下本地安装
pip install "flash_attn-2.7.4.post1+cu124torch2.5.1cxx11abiFALSE-cp311-cp311-
win_amd64.whl"
2、pytorch是cpu版的，重新安装gpu版本
conda install pytorch==2.2.2 torchvision==0.17.2 pytorch-cuda=12.1 -c pytorch
-c nvidia
3、gradio和 opencv 安装
pip install gradio==4.44.1 opencv-python==4.9.0.80
4、numpy降级
pip install numpy==1.26.4
5、安装缺失库
pip install thop psutil ninja
6、pydantic降级
pip install pydantic==2.10.6
cv2 语法修改 cv2.VideoWriter.fourcc 替换掉 cv2.VideoWriter_fourcc 可以不改
8. 发现cv2报错 ImportError: DLL load failed while importing _multiarray_umath: 找
不到指定的模块。 native_module = importlib.import_module("cv2")
ImportError: numpy.core.multiarray failed to import
numpy版本 2.0.2不适⽤，重新安装opencv，numpy降级为1.26.4
pip install numpy==1.26.4

9. gradio启动报错 TypeError: argument of type 'bool' is not iterable，pydantic包
版本问题，回退到2.10.6
pip install pydantic==2.10.6
10. 上传图⽚点击检测报错 pydantic.errors.PydanticSchemaGenerationError: Unable
to generate pydantic-core schema for <class 'starlette.requests.Request'>.
Set arbitrary_types_allowed=True in the model_config to ignore this error or implement
__get_pydantic_core_schema__ on your type to fully support it.
更新 gradio版本
pip install --upgrade gradio==4.44.1
11. yolo_ultralytics 代码包， yolov11-v13集成，整合了视频切分、⾃定义模型推理
a. 图⽚预测
<img width="1814" height="939" alt="image" src="https://github.com/user-attachments/assets/cf94c065-88ba-405f-ac88-d2b8c5fe68ba" />

b. 视频转图⽚集及预测
<img width="1386" height="910" alt="image" src="https://github.com/user-attachments/assets/b96f314d-7937-41d8-a8ab-3b5a8d16220f" />

13. 代码⼯具
a. app.py ⽤于启动界⾯
b. yolov13_transfer.py 把官⽅预训练好的模型参数调整成本地能⽤的，需要将
yolov13{n,s,l,x}.pt都先下载到⽬录下，参数⽂件在这⾥
https://github.com/iMoonLab/yolov13/releases/tag/yolov13，本地代码我已
经改好了不⽤再改了，但后⾯整合代码的时候注意⼀下。
14. 数据集标注⼯具LabelImg下载及使⽤（打开Anaconda Prompt，选择⼀个
python=3.9的环境下）
conda create -n labelimg python=3.9
conda activate labelimg
pip3 install labelImg
labelImg
