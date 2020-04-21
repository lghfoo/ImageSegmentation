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
    types = ['GMMGray', 'GMMColor', 'KMeansGray', 'KMeansColor', 'MGMMGray', 'MGMMColor']
    _fields_ = [
        ("InputImageName", c_char_p),
        ("MaxIterationCount", c_int),
        ("DVarThreshold", c_double),
        ("DExpThreshold", c_double),
        ("DCoeThreshold", c_double),
        ("DLogLikehoodThreshold", c_double),
        ("ComponentCount", c_int),
        ("KMeansInitialized", c_bool),
        ("GMMInitialized", c_bool),
        ("RandomSeed", c_bool),
        ("SegType", c_int),
        ("InputModel", c_char_p),
        ("OutputModel", c_char_p),
        ("Beta", c_double),
        ("WindSize", c_int),
        ("ColorMap", POINTER(GMMColorMap)),
    ]
    def __str__(self):
        return """InputImage\t:{}
K\t\t:{}
DLLThreshold\t:{}
SegType\t\t:{}
KMeansInitialized\t:{}
GMMInitialized\t:{}
RandomSeed\t:{}
Beta\t\t:{}
WindSize\t\t:{}""".format(self.InputImageName.decode('gbk'), 
        self.ComponentCount,
        self.DLogLikehoodThreshold,
        GMMSegArg.types[self.SegType],
        self.KMeansInitialized,
        self.GMMInitialized,
        self.RandomSeed,
        self.Beta,
        self.WindSize)

class GMMSegOutput(Structure):
    _fields_ = [
        ("OutWidth", c_int),
        ("OutHeight", c_int),
        ("OutPixels", POINTER(c_ubyte)),
        ("IterationCount", c_int),
        ("LogLikehoodSummary", POINTER(c_double)),
        ("ModelString", c_char_p)
    ]
    def to_img(self):
        Pixels = (cast(self.OutPixels, POINTER(c_ubyte)))
        img = np.zeros((self.OutHeight, self.OutWidth, 3), np.uint8)
        for i in range(self.OutHeight):
            for j in range(self.OutWidth):
                offset = (i * self.OutWidth + j) * 3
                img[i][j] = tuple(Pixels[offset : offset + 3])
        return img

    def get_LogLikehoodSummary(self):
        Summary = (cast(self.LogLikehoodSummary, POINTER(c_double)))
        Ret = []
        for i in range(self.IterationCount):
            Ret.append(Summary[i])
        return Ret

    def get_ModelString(self):
        return self.ModelString.decode()

# load dll
cur_dir = os.path.dirname(__file__)
os.chdir(cur_dir)
gmm_dll_path = '../x64/Debug/GMM.dll'
assert os.path.exists(gmm_dll_path)
gmm_dll = cdll.LoadLibrary(gmm_dll_path)

# method
gmm_segmentation = gmm_dll.Segmentation
gmm_free_output = gmm_dll.FreeOutput

# call
def create_arg(test_data):
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
    arg.MaxIterationCount = 2048
    arg.KMeansInitialized = test_data[4] # True or False is important for the final result
    arg.GMMInitialized = test_data[5] # True or False is important for the final result
    arg.DLogLikehoodThreshold = 0.001
    arg.ComponentCount = (test_data[1])
    arg.ColorMap = pointer(GMMColorMap(test_data[2]))
    arg.SegType = test_data[3]
    return arg

def create_output():
    output = GMMSegOutput()
    return output

def main():
    output_dir = r"D:\Study\毕业设计\周汇报\第八周\outputs_raw"
    test_datas = [
        # ("D:\\Study\\毕业设计\\周汇报\\第八周\\images\\42049_s0.5_scaled.jpg", 3, [[185, 202, 192], [144, 163, 170], [115, 100, 79]], GMMSegArg.mgmm_color True, True),
        ("D:\\Study\\毕业设计\\周汇报\\第八周\\images\\86016_s0.5_scaled.jpg", 2, [[66, 66, 66], [163, 163, 163]], GMMSegArg.mgmm_gray, True, True),
        ("D:\\Study\\毕业设计\\周汇报\\第八周\\images\\80099_s0.5_scaled.jpg", 2, [[230, 230, 230], [25, 25, 25]], GMMSegArg.mgmm_gray, False, True),
        ("D:\\Study\\毕业设计\\周汇报\\第八周\\images\\374067_s0.5_scaled.jpg", 3, [[123, 123, 123], [36, 36, 36], [74, 74, 74]], GMMSegArg.mgmm_gray, True, True),
        ("D:\\Study\\毕业设计\\周汇报\\第八周\\images\\296059_s0.5_scaled.jpg", 2, [[128, 131, 124], [64, 56, 47]], GMMSegArg.mgmm_color, False, True),
        ("D:\\Study\\毕业设计\\周汇报\\第八周\\images\\113044_s0.5_scaled.jpg", 2, [[176, 204, 117], [173, 118, 98]], GMMSegArg.mgmm_color, True, True),
        ("D:\\Study\\毕业设计\\周汇报\\第八周\\images\\113016_s0.5_scaled.jpg", 2, [[159, 181, 98], [101, 84, 66]], GMMSegArg.mgmm_color, True, True),
        ("D:\\Study\\毕业设计\\周汇报\\第八周\\images\\38092_s0.5_scaled.jpg", 4, [[218, 234, 205], [166, 157, 100], [82, 76, 54], [96, 115, 119]], GMMSegArg.mgmm_color, True, True),
        ("D:\\Study\\毕业设计\\周汇报\\第八周\\images\\310007_s0.5_scaled.jpg", 3, [[185, 202, 192], [144, 163, 170], [115, 100, 79]], GMMSegArg.mgmm_color, True, True),

        ("D:\\Study\\毕业设计\\周汇报\\第八周\\images\\310007_s0.5_scaled.jpg", 3, [[185, 202, 192], [144, 163, 170], [115, 100, 79]], GMMSegArg.gmm_color, True, False),
        ("D:\\Study\\毕业设计\\周汇报\\第八周\\images\\38092_s0.5_scaled.jpg", 4, [[218, 234, 205], [166, 157, 100], [82, 76, 54], [96, 115, 119]], GMMSegArg.gmm_color, True, False),
        ("D:\\Study\\毕业设计\\周汇报\\第八周\\images\\113016_s0.5_scaled.jpg", 2, [[159, 181, 98], [101, 84, 66]], GMMSegArg.gmm_color, True, False),
        ("D:\\Study\\毕业设计\\周汇报\\第八周\\images\\113044_s0.5_scaled.jpg", 2, [[176, 204, 117], [173, 118, 98]], GMMSegArg.gmm_color, True, False),
        ("D:\\Study\\毕业设计\\周汇报\\第八周\\images\\296059_s0.5_scaled.jpg", 2, [[128, 131, 124], [64, 56, 47]], GMMSegArg.gmm_color, False, False),
        ("D:\\Study\\毕业设计\\周汇报\\第八周\\images\\374067_s0.5_scaled.jpg", 3, [[123, 123, 123], [36, 36, 36], [74, 74, 74]], GMMSegArg.gmm_gray, True, False),
        ("D:\\Study\\毕业设计\\周汇报\\第八周\\images\\80099_s0.5_scaled.jpg", 2, [[230, 230, 230], [25, 25, 25]], GMMSegArg.gmm_gray, False, False),
        ("D:\\Study\\毕业设计\\周汇报\\第八周\\images\\86016_s0.5_scaled.jpg", 2, [[66, 66, 66], [163, 163, 163]], GMMSegArg.gmm_gray, True, False),
    ]
    test_data = test_datas[-1]
    arg = create_arg(test_data)

    basename_ext = os.path.basename(arg.InputImageName).decode()
    basename = '{}_{}_{}{}'.format(os.path.splitext(basename_ext)[0].split('_')[0], GMMSegArg.types[arg.SegType], 'K' if arg.KMeansInitialized else 'NK', '_G' if arg.GMMInitialized else '')
    output_dir = os.path.join(output_dir, basename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output = create_output()
    gmm_segmentation(arg, byref(output))
    result = output.to_img()
    cv2.imencode('.png', result)[1].tofile(os.path.join(output_dir, 'result.png'))

    # other file
    ## arg file
    f = open(os.path.join(output_dir, 'arg.txt'), "w")
    f.write(str(arg))
    f.close()
    ## log likehood file
    f = open(os.path.join(output_dir, 'll.txt'), "w")
    f.write('\n'.join([str(l) for l in output.get_LogLikehoodSummary()]))
    f.close()
    ## model file(gmm_gray only)
    f = open(os.path.join(output_dir, 'm.txt'), "w")
    f.write(output.get_ModelString())
    f.close()

    gmm_free_output(byref(output))

main()