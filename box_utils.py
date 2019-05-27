from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image
import xml.etree.ElementTree as ET


def prettyXml(element, indent, newline, level = 0): # elemnt为传进来的Elment类，参数indent用于缩进，newline用于换行
    if element:  # 判断element是否有子元素
        if element.text == None or element.text.isspace(): # 如果element的text没有内容
            element.text = newline + indent * (level + 1)
        else:
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
    temp = list(element) # 将elemnt转成list
    for subelement in temp:
        if temp.index(subelement) < (len(temp) - 1): # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致
            subelement.tail = newline + indent * (level + 1)
        else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个
            subelement.tail = newline + indent * level
        prettyXml(subelement, indent, newline, level = level + 1) # 对子元素进行递归操作


def draw_boxes_on_image(image_path, boxes, scores, labels, label_names, score_thresh=0.5):
    image = np.array(Image.open(image_path))
    plt.figure()
    _, ax = plt.subplots(1)
    ax.imshow(image)

    image_name = image_path.split('/')[-1]
    print("Image {} detect: ".format(image_name))
    colors = {}
    tree = ET.parse('coordinate.xml')
    root = tree.getroot()
    imagename = ET.Element("imgname")
    imagename.attrib = {"name": image_name}
    root.append(imagename)
    for box, score, label in zip(boxes, scores, labels):
        if score < score_thresh:
            continue
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        label = int(label)
        if label not in colors:
            colors[label] = plt.get_cmap('hsv')(label / len(label_names))
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                            fill=False, linewidth=2.0, 
                            edgecolor=colors[label])
        ax.add_patch(rect)
        center = plt.Circle((int(x1 + (x2 - x1)/2), int(y1 + (y2 - y1)/2)), radius=10)
        ax.add_patch(center)
        ax.text(x1, y1, '{} {:.4f}'.format(label_names[label], score), 
                verticalalignment='bottom', horizontalalignment='left',
                bbox={'facecolor': colors[label], 'alpha': 0.5, 'pad': 0},
                fontsize=8, color='white')
        print("\t {:15s} at {:25} score: {:.5f}"
              .format(label_names[int(label)], str(map(int, list(box))), score))
        print("\t {:15s} at  x1:{:.0f}  y1:{:.0f}  x2:{:.0f}  y2:{:.0f}"
              .format(label_names[int(label)], box[0], box[1], box[2], box[3]))
        print("\t {:15s} at  center_x:{}  center_y:{}  "
              .format(label_names[int(label)], int(x1 + (x2 - x1)/2), int(y1 + (y2 - y1)/2)))
        #保存坐标
        labelname = ET.Element("labelname")
        labelname.attrib = {"name": label_names[int(label)]}
        imagename.append(labelname)
        textX = ET.Element("pineappletop_x")
        textX.text = "{}".format(int(x1 + (x2 - x1)/2))
        labelname.append(textX)
        textY = ET.Element("pineappletop_y")
        textY.text = "{}".format(int(y1 + (y2 - y1)/2))
        labelname.append(textY)
        prettyXml(root, '\t', '\n')
        tree.write("coordinate.xml")
    image_name = image_name.replace('jpg', 'png')
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig("./output/{}".format(image_name), bbox_inches='tight', pad_inches=0.0)
    print("Detect result save at ./output/{}\n".format(image_name))
    print("Detect result save at coordinate.xml\n")
    plt.cla()
    plt.close('all')

