
import os
from time import sleep
from datetime import datetime
import socket
import torch
sleep(30)

time_now = datetime.now().strftime("%H:%M:%S")


tensor = torch.rand(3, 3).cuda()
print('Task {}: Hello world from {} at {}.'.format(os.environ["SLURM_PROCID"], socket.gethostname(), time_now))
print("Torch version:", torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)

tensor = torch.rand(3, 3).cuda()

print("Random Tensor:", tensor.device)

print(os.listdir('.'))