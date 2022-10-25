stride = 1
kernel_size = 5
padding = 0
dilation = 1
H_in = 28
depth = 2


for i in range(depth):
    H_out = ((H_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1
    print("H_out: {}, depth: {}" .format(H_out, i))
    H_in = H_out

