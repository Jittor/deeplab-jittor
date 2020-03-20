# Jittor DeepLab

0. 为了更好的运行这份代码，建议先对这份 [教程](<https://cg.cs.tsinghua.edu.cn/jittor/tutorial/2020-3-17-09-55-segmentation/>)  进行阅读。

1. 你可以在下方链接处下载 ImageNet pretrain model。 

[resnet_101](<https://cloud.tsinghua.edu.cn/f/736e24afb94347b8a0b6/?dl=1> )

[resnet_50](<https://cloud.tsinghua.edu.cn/f/312f45548f98476ab29b/?dl=1> )

由于我们暂时没有 ImageNet 的 Pre-train model 所以暂时先使用的是 pytorch 的预训练模型。这里也同时体现了我们和 pytorch 是兼容的。模型参数[来源](<https://drive.google.com/file/d/1NwcwlWqA-0HqAPk3dSNNPipGMF0iS0Zu/view>) 。

2. 把数据集的路径和模型的路径分别替换成本地的数据集路径和模型路径。
3. 运行下面脚本，即可运行。

```shell
sh train.sh
```



 