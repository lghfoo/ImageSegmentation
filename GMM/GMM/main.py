from ctypes import *
import os
import numpy as np
import cv2

c_opencv_dll_dir = r"D:\Software\OpenCV_4.20\PreBuild\opencv\build\x64\vc15\bin"
os.add_dll_directory(c_opencv_dll_dir)

class GMMColor(Structure):
    _fields_ = [
        ("R", c_ubyte),
        ("G", c_ubyte),
        ("B", c_ubyte)
    ]

    def __init__(self, color):
        self.R = color[0]
        self.G = color[1]
        self.B = color[2]
    
    def __str__(self):
        return "[{}, {}, {}]".format(self.R, self.G, self.B)

class GMMColorMap(Structure):
    _fields_ = [
        ("Count", c_int),
        ("Colors",POINTER(GMMColor))
    ]
    def __init__(self, colors):
        self.Count = len(colors)
        self.Colors = (GMMColor * self.Count)()
        for i in range(self.Count):
            self.Colors[i] = GMMColor(colors[i])
    
    def __str__(self):
        colors = []
        for i in range(self.Count):
            colors.append(str(self.Colors[i]))
        return "[{}]".format(", ".join(colors))

class GMMSegArg(Structure):
    gmm_gray = 0
    gmm_color = 1
    kmeans_gray = 2
    kmeans_color = 3
    mgmm_gray = 4
    mgmm_color = 5
    _fields_ = [
        ("InputImageName", c_char_p),
        ("MaxIterationCount", c_int),
        ("DVarThreshold", c_double),
        ("DExpThreshold", c_double),
        ("DCoeThreshold", c_double),
        ("DLogLikehoodThreshold", c_double),
        ("ComponentCount", c_int),
        ("KMeansInitialized", c_bool),
        ("RandomSeed", c_bool),
        ("SegType", c_int),
        ("InputModel", c_char_p),
        ("OutputModel", c_char_p),
        ("Beta", c_double),
        ("WindSize", c_int),
        ("ColorMap", POINTER(GMMColorMap)),
    ]

class GMMSegOutput(Structure):
    _fields_ = [
        ("OutWidth", c_int),
        ("OutHeight", c_int),
        ("OutPixels", POINTER(c_ubyte))
    ]
    def to_img(self):
        Pixels = (cast(self.OutPixels, POINTER(c_ubyte)))
        img = np.zeros((self.OutHeight, self.OutWidth, 3), np.uint8)
        for i in range(self.OutHeight):
            for j in range(self.OutWidth):
                offset = (i * output.OutWidth + j) * 3
                img[i][j] = tuple(Pixels[offset : offset + 3])
        return img

# load dll
cur_dir = os.path.dirname(__file__)
os.chdir(cur_dir)
gmm_dll_path = '../x64/Debug/GMM.dll'
assert os.path.exists(gmm_dll_path)
gmm_dll = cdll.LoadLibrary(gmm_dll_path)

# method
gmm_segmentation = gmm_dll.Segmentation


test_datas = [
    ("D:\\Study\\毕业设计\\周汇报\\第八周\\images\\80099_s0.5_scaled.jpg", 2, [[230, 230, 230], [25, 25, 25]]),
    ("D:\\Study\\毕业设计\\周汇报\\第八周\\images\\310007_s0.5_scaled.jpg", 3, [[185, 202, 192], [144, 163, 170], [115, 100, 79]]),
]

# call
def create_arg():
    test_data = test_datas[-1]
    arg = GMMSegArg()
    arg.InputImageName = c_char_p(str.encode(test_data[0], encoding='gbk'))
    arg.RandomSeed = False
    arg.Beta = 12
    arg.WindSize = 5
    arg.InputModel = None
    arg.DCoeThreshold = 0.001
    arg.DExpThreshold = 0.001
    arg.DVarThreshold = 0.001
    arg.OutputModel = None
    arg.MaxIterationCount = 1024
    arg.KMeansInitialized = True
    arg.DLogLikehoodThreshold = 0.001
    arg.ComponentCount = (test_data[1])
    arg.ColorMap = pointer(GMMColorMap(test_data[2]))
    arg.SegType = GMMSegArg.mgmm_color
    return arg

def create_output():
    output = GMMSegOutput()
    return output

arg = create_arg()
output = create_output()
gmm_segmentation(arg, byref(output))

result = output.to_img()
cv2.imencode('.png', result)[1].tofile(os.path.join(cur_dir, 'test.png'))
