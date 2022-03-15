# Jittor 第二届草图生成风景比赛 baseline

## z dataset

```
git clone git@github.com:zhouwy19/gan-baseline.git
cd gan-baseline
wget 
unzip jittor_landscape_100k_dataset.zip
```

## Requirements

```
jittor
pillow
opencv-python
```

## Train

单卡训练
```
bash scripts/single_gpu.sh
```

多卡训练
```
bash scripts/multi_gpu.sh
```