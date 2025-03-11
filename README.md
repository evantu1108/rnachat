# RNA Function Prediction

## Getting Started
### Installation

**1. Prepare the the environment**


```bash
cd RNAChat
conda env create -f rnachat.yml
conda activate rnachat
```


**2. Prepare the pretrained Vicuna weights**

```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash;
sudo apt-get install git-lfs;
git clone https://huggingface.co/lmsys/vicuna-13b-v1.5;
```

**3. Prepare RNA encoder**

```bash
git clone https://github.com/lbcb-sci/RiNALMo.git; 
```
