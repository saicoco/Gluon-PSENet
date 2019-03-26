# PSENet

PSENet reproduce
### Implements  

#### dataloader  
数据增强部分参考至[issue10](https://github.com/whai362/PSENet/issues/10), 对于短边低于640的图片，短边缩放至640;大于640的不惊醒缩放。然后进行随机crop

#### Network  

模型部分参考至[gluoncv_model_zoo](https://gluon-cv.mxnet.io/model_zoo/classification.html)中**resnet50_v1s**,其他pretrain_model可以酌情替换，默认模型下载路径为`~/.mxnet/`下  

#### Loss  
loss部分直接使用dice_coefficient, OHEM部分实现存在BUG，需进一步修改。可以使用的损失函数：  
- [Generalized-IOU](https://github.com/generalized-iou)
- weighted binart cross entropy
- etc.  

#### Inference  

预测阶段，借鉴[issue 15](https://github.com/whai362/PSENet/issues/15), 当然这里有更好的实现[tensorflow_PSENET](https://github.com/liuheng92/tensorflow_PSENet)  

### Usage  

#### Train  

```
python scripts/train.py $data_path $ckpt
``` 
- `data_path`表示数据路径，存放a.jpg, a.txt.注意前缀需要一致  
- `ckpt`即pretrain-model, 默认是`~/.mxnet/xxx.params`  

#### Inference  
```
python scripts/eval.py $data_path $ckpt
```  
这部分存在bug，需进一步进行修复

### TODO
- 完成OHEM部分


### result:
损失函数正常，只是kernel结果图的显示存在问题
![img](loss.png)
