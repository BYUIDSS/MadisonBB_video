import torch

data = torch.randn(10, 10, 3)

# make a tensor with random numbers
tensor_rand = torch.rand(5,3)
print(tensor_rand)

# make a tensor filled with zeros that are of long type
tensor1 = torch.empty(5,3, dtype=torch.long)
print(tensor1)

# construct a tensor directly from data
tensordata = torch.tensor(data)
print(tensordata)

# create a tensor filled with ones
tensor_one = torch.ones(5, 5)
print(tensor_one)

# create tensor based off of an old one
tensor_new = torch.randn_like(tensordata, dtype= torch.float32)
print(tensor_new)
# get its size
tensor_size = tensor_new.size()
print(tensor_size)
print(data)

x = torch.randint(low=0, high=10, size=(5, 3))

# put our tensors on the gpu
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))
