# 中文使用说明

本版本提供 CMPANet-S 的训练和测试代码。地图编码器为 ViT-Small，冻结影像编码器为 ViT-Base，交叉注意力头数为24。训练、测试均采用**四个特征块共享同一个 norm1**。

## 已核验精度

`cmpanet_small_head_24_max_performance.pth` 在完整397张验证集上的结果：Precision **85.22%**、Recall **78.47%**、F1 **81.70%**、变化类IoU **69.07%**、OA **99.12%**。

整理后的 `test.py` 与此前独立验证的混淆矩阵完全相同。这是已有权重复测，不能称为当前训练代码从零训练复现论文。历史完整源码未找回，本实现依据现存源码与权重做严格加载和兼容重建；不能仅凭指标接近推断历史实现完全相同。

## 环境与权重

安装命令见[主README](../README.md#installation)。需要 `ssl_vit_base_epoch8.pth` 和 `cmpanet_small_head_24_max_performance.pth` 两个文件，前者为冻结影像编码器，后者为变化检测网络和地图编码器。

训练好的 `cmpanet_small_head_24_max_performance.pth` 已提供[百度网盘下载](https://pan.baidu.com/s/15o1z2YPFryg5aEJ_Chi6NA)，提取码：`cmcd`。预训练影像编码器 `ssl_vit_base_epoch8.pth` 的链接仍待上传，完整测试需要这两个文件。请按[下载表](../README.md#model-downloads)中的SHA-256核对；其他目录的同名权重不一定是这一份。

## 数据

训练5225对、验证397对，每幅512×512。清单每行依次为地图、影像、标签三个路径，以空格分隔；样本路径不能含空格，相对路径以执行命令的目录为准。使用原有划分。

```text
D:/dataset/Hlubm/0001.png D:/dataset/Crism/0001.png D:/dataset/Target/0001.png
```

OpenCV保持BGR顺序；地图以三通道读取。归一化参数在配置中。原标签255代表变化。地图全0或影像全0/全1像素被忽略。有效像素99,373,806，忽略像素4,697,362。变化类IoU与mIoU分开报告。

## 测试

在仓库根目录执行：

```bash
python test.py --checkpoint checkpoints/cmpanet_small_head_24_max_performance.pth --pretrained pretrained/ssl_vit_base_epoch8.pth --valtxt data/val.txt --output outputs/small24 --expect-samples 397
```

输出精度JSON、逐图混淆矩阵、评估配置。增加 `--save-predictions` 保存预测图，每次使用新输出目录。测试不要求训练集。

## 训练

```bash
python train.py --suptxt data/train.txt --valtxt data/val.txt --pretrained pretrained/ssl_vit_base_epoch8.pth --epochs 100 --train_batch_size 4 --base_lr 0.00001 --test_interval 500 --checkpoints_name cmpanet_small_head24
```

当前方案为 OHEM CE + KL(student_map || frozen_image_teacher)、共享norm1、AdamW、AMP、batch=4、100轮。历史权重使用的分割损失、热启动与当前方案存在差异，历史patch loss公式未确证，未承诺新训练必然获得81.70%。

`--resume` 只加载网络权重做微调，不恢复优化器、AMP或随机数状态。`--check-config` 只检查配置与路径，`--memory-fraction` 默认0.60。重复每batch的参数量日志已移除。

`scripts/` 提供 `.sh`、`.ps1` 包装脚本。论文图是原论文结果，不是本次新生成预测。其他参数及证据见[主README](../README.md)和[验证记录](VERIFICATION.md)。
