'''
Description: 二值化
Author: Cyletix
Date: 2023-03-17 21:26:19
LastEditTime: 2023-04-02 23:29:52
FilePath: \AutoMakeosuFile\binarize.py
'''
# from PIL import Image
# import pytesseract


def simple_binarize(chroma0):
    import copy
    chroma=copy.deepcopy(chroma0)#深度拷贝,不修改原变量
    threshold=0.9 #阈值
    for i in range(len(chroma)):
        for j in range(len(chroma[0])):
            if chroma[i][j]>threshold:
                chroma[i][j]=1
            else:
                chroma[i][j]=0
    return chroma

# def read_text(text_path):
#     """
#     传入文本(jpg、png)的绝对路径,读取文本
#     :param text_path:
#     :return: 文本内容
#     """
#     # 验证码图片转字符串
#     im = Image.open(text_path)
#     # 转化为8bit的黑白图片
#     imgry = im.convert('L')
#     # 二值化，采用阈值分割算法，threshold为分割点
#     threshold = 140
#     table = []
#     for j in range(256):
#         if j < threshold:
#             table.append(0)
#         else:
#             table.append(1)
#     out = imgry.point(table, '1')
#     # 识别文本
#     text = pytesseract.image_to_string(out, lang="eng", config='--psm 6')
#     return text

# %%

# 图像二值化
def threshold(self):
    import cv2 as cv
    src = self.cv_read_img(self.src_file)
    if src is None:
        return

    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    # 这个函数的第一个参数就是原图像，原图像应该是灰度图。
    # 第二个参数就是用来对像素值进行分类的阈值。
    # 第三个参数就是当像素值高于（有时是小于）阈值时应该被赋予的新的像素值
    # 第四个参数来决定阈值方法，见threshold_simple()
    # ret, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    ret, dst = cv.threshold(gray, 127, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    self.decode_and_show_dst(dst)



# %%
# def local_binarize(image):
#     import cv2
#     import numpy as np

#     # 读取输入图像
#     input_image = cv2.imread(image, 0)

#     # 定义局部二值化参数
#     block_size = (3, 25)
#     constant = 2

#     # 应用局部二值化
#     output_image = cv2.adaptiveThreshold(input_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, constant)

#     # 显示输出图像
#     cv2.imshow('Output Image', output_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()








# %%
if __name__=='__main__':
    ndarray_test=[[1,2,3],[3,2,1],[1,3,2]]

    local_binarize('Back.png')


    import cv2
    # 定义局部二值化参数
    block_size = 3
    constant = 2

    #图片输入
    image='E:\osu!\Songs\DJ Genki VS Camellia feat moimoi - YELL! [6k]\Back.png'
    input_image = cv2.imread(image,0)
    output_image = cv2.adaptiveThreshold(input_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, constant)
    cv2.imshow('Output Image', output_image)



    #向量输入
    input_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # 先转换为二值化期望的输入:灰度图
    gray_array = cv2.cvtColor(input_array, cv2.COLOR_BGR2GRAY)
    # 应用局部二值化

    # 显示输出图像


    #映射到rgb值域
    chroma1 = (chroma*255).astype('uint8')
    threshold = cv2.adaptiveThreshold(chroma1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,2)
    plt.imshow(threshold,'gray')
