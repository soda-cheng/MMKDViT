# MMKDViT
## **Paper: Multi-Layer Multi-Knowledge Distillation for Vision Transformers**

![](D:\git-supervise\mmkdvit\MMKDViT\docs\whole.jpg)

Vision Transformers (ViTs) have shown strong representation capability by modeling long-range dependencies through self-attention. However, ViTs performance often degrades when large-scale labeled data are limited. Knowledge distillation offers an effective solution by transferring teacher knowledge to lightweight students, yet existing methods mostly focus on either final logits or single-stage intermediate features, overlooking the hierarchical structural knowledge and class-specific information distributed across Transformer layers. 

To address this limitation, we propose Multi-Layer Multi-Knowledge Distillation for ViTs (MMKDViT),  a unified framework that transfers both spatial structure and class-level semantics in a layer-aware manner. Specifically, MMKDViT introduces Multi-Layer Attention Distillation to preserve token dependency structures and Multi-Layer Decoupled Dlass Distillation to separately supervise primary and non-primary class knowledge, thereby improving feature selectivity and class discrimination.  Experiments on datasets like ImageNet-1k, CIFAR100, and CUB-2011 demonstrate the effectiveness of MMKDViT, outperforming existing ViTs distillation methods and achieving superior results.

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



## Citation

```
@article{yi2026MMKDViT,
  title={Multi-Layer Multi-Knowledge Distillation for Vision Transformers},
  author={Jianping Gou, Cheng Yi, Lan Du, Yujun Cai, Zhang Yi, Dacheng Tao},
  year={2026}
}
```

