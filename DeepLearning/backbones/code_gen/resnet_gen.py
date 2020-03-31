
def gen(code):
    print(code)

def gen_conv_def(var, i, o, k, s=1, p=1):
    code = "{}nn.Conv2d(in_channels={}, out_channels={}, kernel_size={}, stride={}, padding={})".format(
        var, i, o, k , s, p
    )
    gen(code)

def gen_max_pool_def(var, k, s):
    code = "{}nn.MaxPool2d(kernel_size={}, stride={})".format(var, k, s)
    gen(code)

def gen_relu_def(var):
    code = "{}nn.ReLU()"
    gen(code)
