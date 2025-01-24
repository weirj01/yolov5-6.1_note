# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (MacOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

'''===============================================一、导入包==================================================='''
'''====================================1.导入安装好的python库========================================'''
import argparse # 解析命令行参数的库
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

'''==================================================2.获取当前文件的绝对路径===================================================='''
FILE = Path(__file__).resolve()  # __file__是当前文件(即detect.py),resolve()获取绝对路径,比如D://yolov5/detect.py
ROOT = FILE.parents[0]  # YOLOv5 root directory  ROOT保存着当前项目的父目录,比如 D://yolov5
if str(ROOT) not in sys.path:  # sys.path即当前python环境可以运行的路径,假如当前项目不在该路径中,就无法运行其中的模块,所以就需要加载路径
    sys.path.append(str(ROOT))  # add ROOT to PATH  把ROOT添加到运行路径上
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative ROOT设置为相对路径

'''==================================================3..加载自定义模块===================================================='''
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

'''==================================================二、run函数——传入参数===================================================='''
 
'''====================================1.载入参数========================================'''
@torch.no_grad()# 该标注使得方法中所有计算得出的tensor的requires_grad都自动设置为False，也就是说不进行梯度的计算(当然也就没办法反向传播了)， 节约显存和算
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)    模型权重文件
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam 测试文件路径
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path   标签路径？？
        imgsz=(640, 640),  # inference size (height, width) 缩放后的输入yolo的图片大小
        conf_thres=0.25,  # confidence threshold    置信度阈值， 高于此值的bounding_box才会被保留，NMS
        iou_thres=0.45,  # NMS IOU threshold    NMS的IOU阈值
        max_det=1000,  # maximum detections per image      单图片最大检测数量
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu

        view_img=False,  # show results     结果展示
        save_txt=False,  # save results to *.txt    结果保存，在路径runs/detect/exp*
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes   保存裁剪后的预测框
        nosave=False,  # do not save images/videos

        classes=None,  # filter by class: --class 0, or --class 0 2 3 过滤指定类的预测结果
        agnostic_nms=False,  # class-agnostic NMS   NMS去除不同类别的框？？
        augment=False,  # augmented inference TTA测试时增强/多尺度预测，可以提分？
        visualize=False,  # visualize features  可视化网络特征
        update=False,  # update all models  如果为True,则对所有模型进行strip_optimizer操作,去除pt文件中的优化器等信息？？
        
        project=ROOT / 'runs/detect',  # save results to project/name 预测结果保存的路径
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)    绘制bbox的线宽
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference    半精度推理
        dnn=False,  # use OpenCV DNN for ONNX inference    使用OpenCV DNN进行ONNX推理？？
        ):
        
    '''====================================2.初始化配置========================================'''
    source = str(source)# 将输入路径source转换为字符串。
    save_img = not nosave and not source.endswith('.txt')  # save inference images
     # 判断source是不是视频/图像文件路径
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    # 判断source是否是链接
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
     # 判断是source是否是摄像头
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download # 返回文件。如果source是一个指向图片/视频的链接,则下载输入数据

    '''====================================3.保存结果========================================'''
    # Directories
    # save_dir是保存运行结果的文件夹名，是通过递增的方式来命名的。第一次运行时路径是“runs\detect\exp”，第二次运行时路径是“runs\detect\exp1”
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    '''====================================4.加载模型========================================'''
    # Load model
    device = select_device(device)# 获取设备 CPU/CUDA
    # DetectMultiBackend定义在models.common模块中，是我们要加载的网络，其中weights参数就是输入时指定的权重文件（比如yolov5s.pt）
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    '''
        stride：推理时所用到的步长，默认为32， 大步长适合于大目标，小步长适合于小目标
        names：保存推理结果名的列表，比如默认模型的值是['person', 'bicycle', 'car', ...] 
        pt: 加载的是否是pytorch模型（也就是pt格式的文件）
        jit：当某段代码即将第一次被执行时进行编译，因而叫“即时编译”
        onnx：利用Pytorch我们可以将model.pt转化为model.onnx格式的权重，在这里onnx充当一个后缀名称，
              model.onnx就代表ONNX格式的权重文件，这个权重文件不仅包含了权重值，也包含了神经网络的网络流动信息以及每一层网络的输入输出信息和一些其他的辅助信息。
    '''
    #stride步长是什么意思，卷积滑窗步长？？
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size # 确保输入图片的尺寸imgsz能整除stride=32 如果不能则调整为能被整除并返回

    # Half
    # 如果不是CPU，使用半精度(图片半精度/模型半精度)。半精度只支持cuda ？？
    half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    '''====================================5.加载数据========================================'''
    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        #加载一个batch_size的数据流？ 得到的结果是什么？为何还要pt参数？
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size  为何是1？？
    vid_path, vid_writer = [None] * bs, [None] * bs

    '''====================================6.推理部分========================================'''
    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz), half=half)  # warmup
    # dt: 存储每一步骤的耗时，seen: 统计已经处理的图片数量
    dt, seen = [0.0, 0.0, 0.0], 0
    # 遍历数据集，进行推理
    # path: 图片路径，im: resize后的图片，im0s: 原始图片，vid_cap: 视频流，s: 图片基本信息,路径、大小等
    for path, im, im0s, vid_cap, s in dataset:
        #预处理
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)# 将图片放到指定设备(如GPU)上识别。#torch.size=[3,640,480]
        im = im.half() if half else im.float()  # uint8 to fp16/32 # 把输入图片从整型转化为半精度/全精度浮点数？？
        im /= 255  # 0 - 255 to 0.0 - 1.0 归一化，所有像素点除以255
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim 添加一个第0维。扩充batch尺寸到im 变成[1，3,640,480]？？
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        # 可视化文件路径。
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        '''=====推理====='''
        # 推理结果，pred保存的是所有的bound_box的信息，torch.size=[1,18900,85]？？
        pred = model(im, augment=augment, visualize=visualize)#augment：是否使用数据增强，visualize：是否可视化
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS ，返回值为过滤后的预测框
        # conf_thres置信度阈值；iou_thres iou阈值； classes保留特定类别，默认none；agnostic_nms,去除不同类别的框； max_det=max_det 最大检测数量
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count #取出dataset中的一张图片
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path 当前路径yolov5/data/images/
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh 得到原图的宽和高？？
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))# 得到一个绘图的类，类中预先存储了原图、线条宽度、类名
            # 判断有没有框
            if len(det):
                # Rescale boxes from img_size to im0 size 将预测框信息映射到原图 位置（经过了缩放）
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results 保存预测结果
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    '''====================================7.在终端里打印出运行的结果========================================'''
    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


# 命令使用
# python detect.py --weights runs/train/exp_yolov5s/weights/best.pt --source  data/images/fishman.jpg # webcam
if __name__ == "__main__":
    opt = parse_opt() # 解析参数
    main(opt) # 执行主函数
