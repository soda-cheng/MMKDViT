# GFRCKD
## **Paper: Vision Transformer-Based Generated Feature Distillation Using Recycled Class-Token for Visual Recognition**

![](D:\KYYC\GFRCKD论文内容\图片\readme_0.png)

Vision Transformer (ViT) has recently achieved remarkable results across various visual tasks, leveraging its unique multi-heads self-attention mechanism. However, the high-performing ViT models often possess large token dimensions, deep network depths, and an enormous number of parameters, rendering them impractical for real-world applications. In contrast, lightweight ViT models with smaller token dimensions struggle to learn comprehensive feature knowledge. Knowledge distillation has been proposed as a model compression technique to transfer rich knowledge in large ViT models to compact ones. Current distillation methods for ViT models primarily focus on logits-based distillation, generally overlooking intermediate feature knowledge and classification knowledge within the class-token. 

In response, this paper proposes an efficient distillation method named Generated Feature with Recycled Class-Token Knowledge Distillation (GFRCKD). Specifically, we recycle the class token from the pre-trained teacher model after dimension conversion, and incorporate it into the student model to form a new classifier for inference.Due to significant differences in the deep layer attention of ViT models, it is challenging for the student to directly mimic the teacher final features.

## Train

```
#multi GPU
bash tools/dist_train.sh configs/distillers/imagenet/deit-s3_distill_deit-t_img.py 4
python -m torch.distributed.launch tools/train.py --cfg configs/deit/deit-small_pt-4xb256_in1k.py --data-path
```

## Transfer

```
# Tansfer the Distillation model into mmcls model
python pth_transfer.py --dis_path $dis_ckpt --output_path $new_mmcls_ckpt
```

## Test

```
#multi GPU
bash tools/dist_test.sh configs/deit/deit-tiny_pt-4xb256_in1k.py $new_mmcls_ckpt 8 --metrics accuracy
```

## Results

<img src="D:\KYYC\GFRCKD论文内容\图片\deit_imagenet.png" style="zoom:50%;" />

## Citation

```
@article{yi2026gfrckd,
  title={Vision Transformer-Based Generated Feature Distillation Using Recycled Class-Token for Visual Recognition},
  author={Jianping Gou, Cheng Yi, Liyuan Sun, Lan Du, Xin Luo and Yi Zhang},
  year={2026}
}
```

