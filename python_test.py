
import os
from time import sleep
from datetime import datetime
import socket
import torch
sleep(30)

time_now = datetime.now().strftime("%H:%M:%S")

print('Task {}: Hello world from {} at {}.'.format(os.environ["SLURM_PROCID"], socket.gethostname(), time_now))
print("Torch version:", torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)