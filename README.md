

# Installation

To set-up python 3.10, Torch 2.3.0 and Cuda 11.8 (requirements for gsplat and kaolin)
```
conda create -n vsres310 python=3.10 -y
conda activate vsres310

conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

conda install -c nvidia -c pytorch -c conda-forge \
    pytorch=2.3.0 torchvision=0.18.0 torchaudio=2.3.0 \
    pytorch-cuda=11.8 -y

pip install -r requirements.txt

cd submodules/
git clone https://github.com/fbcotter/pytorch_wavelets
cd pytorch_wavelets
pip install .

cd ../../

# GSplat 1.5.3 wheel for pt2.3.0 and cu11.8
pip install gsplat --index-url https://docs.gsplat.studio/whl/pt23cu118

# Kaolin 0.18 wheel for pt2.3.0 and cu11.8
pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.3.0_cu118.html

```