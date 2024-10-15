import cv2
import numpy as np
import torch.utils.data as data
from PIL import Image
from PIL import ImageOps
from PIL import ImageEnhance
Image.LOAD_TRUNCATED_IMAGES = True # 防止图片过大，加载不完整



def preprocess_input(x):
    x /= 127.5
    x -= 1.
    return x


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[-2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

class DataGenerator(data.Dataset):
    def __init__(self, annotation_lines, inpt_shape, random=True, window_size=5):
        self.window_size = window_size
        self.targets = [int(line.split(';')[0]) for line in annotation_lines]  # 收集所有标签
        self.annotation_lines = annotation_lines
        self.input_shape = inpt_shape
        self.random = random

    def __len__(self):
        return len(self.annotation_lines)

    def __getitem__(self, index):
        annotation_path = self.annotation_lines[index].split(';')[1].split()[0]
        image = Image.open(annotation_path)
        image = self.get_random_data(image, self.input_shape, index, random=self.random)
        image = np.transpose(preprocess_input(np.array(image).astype(np.float32)), [2, 0, 1])
        y = int(self.annotation_lines[index].split(';')[0])
        return image, y, annotation_path

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, inpt_shape, index, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):

        image = cvtColor(image)
        iw, ih = image.size
        h, w = inpt_shape
        if not random:
            scale = min(w / iw, h / ih) # 缩放比例
            nw = int(iw * scale) # 新的宽度
            nh = int(ih * scale) # 新的高度
            dx = (w - nw) // 2 # x方向的偏移量
            dy = (h - nh) // 2 # y方向的偏移量

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))

            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)
            return image_data
        new_ar = w / h * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter) # 随机长宽比
        scale = self.rand(.75, 1.25)  # 随机缩放
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        #将图像多余的部分加上灰条
        # dx=int(self.rand(0,w-nw)) #x方向的偏移量
        # dy=int(self.rand(0,h-nh)) #y方向的偏移量
        dx = (w - nw) // 2  # x方向的偏移量
        dy = (h - nh) // 2  # y方向的偏移量
        new_image=Image.new('RGB',(w,h),(128,128,128))
        new_image.paste(image,(dx,dy))
        image=new_image




        # # 随机缩小
        # shrink_factor_w = self.rand(0.5, 1)  # 宽度缩小因子
        # shrink_factor_h = self.rand(0.5, 1)  # 高度缩小因子
        # # 计算新尺寸
        # nw, nh = int(iw * shrink_factor_w), int(ih * shrink_factor_h)
        # image = image.resize((nw, nh), Image.BICUBIC)
        # # 计算填充量
        # pad_w = (w - nw) // 2
        # pad_h = (h - nh) // 2
        # new_image = Image.new('RGB', (w, h), (128, 128, 128))
        # new_image.paste(image, (pad_w, pad_h))
        # image = new_image

        # # 随机噪声
        # if self.rand() < 0.5:
        #     image_np = np.array(image)  # 将图像转换为numpy数组
        #     noise = np.random.normal(0, 0.02, (h, w, 3))
        #     image_np = image_np + noise
        #     image = Image.fromarray(image_np.astype('uint8'))  # 将图像转换回PIL Image对象
        #
        # # 随机亮度、对比度调整
        # if self.rand() < 1:
        #     brightness_factor = self.rand(0.8, 1.2)
        #     image = ImageEnhance.Brightness(image).enhance(brightness_factor)
        # if self.rand() < 1:
        #     contrast_factor = self.rand(0.8, 1.2)
        #     image = ImageEnhance.Contrast(image).enhance(contrast_factor)

        # 垂直翻转
        if self.rand() < .5:
            image = ImageOps.flip(image)
        # 翻转图像
        flip = self.rand() < .5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT) 
        rotate = self.rand() < .5
        if rotate:
            angle = np.random.randint(-30, 30)
            a, b = w / 2, h / 2
            M = cv2.getRotationMatrix2D((a, b), angle, 1)
            image = cv2.warpAffine(np.array(image), M, (w, h), borderValue=[128, 128, 128])  # 旋转图像

        image_data = np.array(image, np.float32)




        return image_data
