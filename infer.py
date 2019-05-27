import os
import numpy as np
import paddle.fluid as fluid
import box_utils
import reader
from models.yolov3 import YOLOv3
import xml.etree.ElementTree as ET

def infer():
    use_gpu = False
    input_size = 608

    model = YOLOv3(is_train=False)
    model.build_model()
    outputs = model.get_pred()
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    if "./weights/model_iter53999":
        def if_exist(var):
            return os.path.exists(os.path.join("./weights/model_iter53999", var.name))
        fluid.io.load_vars(exe, "./weights/model_iter53999", predicate=if_exist)

    feeder = fluid.DataFeeder(place=place, feed_list=model.feeds())
    fetch_list = [outputs]
    image_names = []

    for image_name in os.listdir("./img/"):
        if image_name.split('.')[-1] in ['jpg', 'png']:
            image_names.append(image_name)

    for image_name in image_names:
        infer_reader = reader.infer(input_size, os.path.join("./img/", image_name))
        label_names, _ = reader.get_label_infos()
        data = next(infer_reader())
        outputs = exe.run(
            fetch_list=[v.name for v in fetch_list],
            feed=feeder.feed(data),
            return_numpy=False)
        bboxes = np.array(outputs[0])
        if bboxes.shape[1] != 6:
            print("No object found in {}".format(image_name))
            continue
        labels = bboxes[:, 0].astype('int32')
        scores = bboxes[:, 1].astype('float32')
        boxes = bboxes[:, 2:].astype('float32')
        draw_thresh = 0.40
        path = os.path.join("./img/", image_name)
        box_utils.draw_boxes_on_image(path, boxes, scores, labels, label_names, draw_thresh)


if __name__ == '__main__':
    # 创建根节点
    root = ET.Element("coordinate")
    # 创建elementtree对象，写文件
    tree = ET.ElementTree(root)
    tree.write("coordinate.xml")
    infer()
