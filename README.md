# PSENet

PSENet reproduce by Gluon

### Implements  
- `DiceLoss_With_OHEM`
- `FPN`

#### Network  

- [gluoncv_model_zoo](https://gluon-cv.mxnet.io/model_zoo/classification.html):**resnet50_v1b**, you can replace it with others，the default path of pretrained-model in `~/.mxnet/`

#### Inference  
 

### Usage  

#### Train  

```
python scripts/train.py $data_path $ckpt
``` 
- `data_path`: path of dataset, which the prefix of image and annoation must be same, for example, a.jpg, a.txt  
- `ckpt`: the filename of pretrained-mdel  

#### Inference  
```
python scripts/eval.py $data_path $ckpt
``` 
### References  
- [issue 15](https://github.com/whai362/PSENet/issues/15), 
- [tensorflow_PSENET](https://github.com/liuheng92/tensorflow_PSENet) 
- [issue10](https://github.com/whai362/PSENet/issues/10)

