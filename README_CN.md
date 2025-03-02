# DiffModeler

<a href="https://github.com/marktext/marktext/releases/latest">
   <img src="https://img.shields.io/badge/DiffModeler-v1.0.0-green">
   <img src="https://img.shields.io/badge/platform-Linux%20%7C%20Mac%20-green">
   <img src="https://img.shields.io/badge/Language-python3-green">
   <img src="https://img.shields.io/badge/dependencies-tested-green">
   <img src="https://img.shields.io/badge/licence-GNU-green">
</a>  

DiffModeler是一个计算工具，使用扩散模型从0-20Å分辨率的冷冻电镜图中自动构建完整的蛋白质复合物结构。

版权所有 (C) 2023 王晓、朱晗、寺师元基、木原大辅和普渡大学。

许可证：GPL v3。（如果您对其他许可证感兴趣，例如商业用途，请联系我们。）

联系方式：
- Daisuke Kihara (dkihara@purdue.edu)
- 技术问题请联系：Xiao Wang (wang3702@purdue.edu)

## 引用：

Xiao Wang, Han Zhu, Genki Terashi, Manav Taluja, & Daisuke Kihara. DiffModeler: large macromolecular structure modeling for cryo-EM maps using a diffusion model. Nature Methods, 2024. [论文链接](https://www.nature.com/articles/s41592-024-02479-0)

```
@article{wang2024diffmodeler,   
  title={DiffModeler: Large Macromolecular Structure Modeling for Cryo-EM Maps Using a Diffusion Model},   
  author={Xiao Wang, Han Zhu, Genki Terashi, Manav Taluja, and Daisuke Kihara},    
  journal={Nature Methods},    
  year={2024}    
}   
```

## 在线服务：
### 输入map+单链结构：https://em.kiharalab.org/algorithm/DiffModeler
### 输入map+序列：https://em.kiharalab.org/algorithm/DiffModeler(seq)
### 蛋白质+DNA复合物结构建模：https://em.kiharalab.org/algorithm/ComplexModeler

## 简介

冷冻电子显微镜（cryo-EM）已被广泛用于实验中确定多链蛋白质复合物，但当分辨率降低时，建模精度会大大降低。在5-10Å的中等分辨率下，即使是基于模板的结构拟合也面临重大挑战。为解决这个问题，我们引入了DiffModeler，这是一个全自动的蛋白质复合物结构建模方法，它利用扩散模型进行骨架追踪，并使用AlphaFold预测的单链结构进行结构拟合。

在中等分辨率冷冻电镜图的广泛测试中，DiffModeler展示了非常准确的结构建模能力，显著超越了现有方法。值得注意的是，我们成功建模了一个由47条链组成的蛋白质复合物（包含13,462个残基），获得了令人印象深刻的0.9 TM-Score。我们还进一步对10-20Å低分辨率图进行了基准测试，并验证了其泛化能力。

## 整体流程

1) 通过扩散模型从中等分辨率冷冻电镜图中进行骨架追踪
2) 使用AlphaFold进行单链结构预测
3) 使用VESPER进行单链结构拟合
4) 通过组装算法进行蛋白质复合物建模

## 安装

### 系统要求
- **CPU**：4核或更高
- **内存**：12GB RAM或更高
- **GPU**：CUDA兼容，最小12GB显存
- **注意**：GPU是必需的，因为DiffModeler在GPU上执行大部分计算

### 安装步骤
1. 安装git
2. 克隆仓库：
```bash
git clone git@github.com:kiharalab/DiffModeler.git && cd DiffModeler
```

3. 配置DiffModeler环境
#### 方案A：Conda环境
1. 从 https://www.anaconda.com/download#downloads 安装anaconda
2. 通过yml文件安装环境：
```bash
conda env create -f environment.yml
```
3. 激活运行环境：
```bash
conda activate DiffModeler
```

### 4. 下载预训练模型
扩散模型权重（在5-10Å上训练，可用于2-5Å（非常好的骨架追踪）和10-20Å）：[diffusion_model](https://huggingface.co/zhtronics/DiffModelerWeight/resolve/main/diffusion_best.pth.tar)

您也可以使用命令行下载：
```commandline
mkdir best_model
cd best_model
wget https://huggingface.co/zhtronics/DiffModelerWeight/resolve/main/diffusion_best.pth.tar
```

## 使用方法

### 1. 使用单链结构进行蛋白质复合物建模

可以通过以下命令运行：
```bash
python3 main.py --mode=0 -F=[Map_Path] -P=[Structure_Path] -M=[Info_Path] --config=[pipeline_config_file] --contour=[Contour_Level] --gpu=[GPU_ID] --resolution=[resolution]
```

参数说明：
- [Map_Path]：输入的实验冷冻电镜图路径
- [Structure_Path]：单链结构文件目录或zip文件路径
- [Info_Path]：输入信息文件路径
- [pipeline_config_file]：流程参数配置文件，保存在`config`目录中
- [Contour_Level]：图密度阈值，用于移除外部区域以节省处理时间（建议使用作者推荐轮廓水平的一半）
- [GPU_ID]：用于推理的GPU
- [resolution]：图分辨率，0-2Å将跳过扩散模型，因此这里可以使用近似值

您有两种方式提供单链结构：
1. 使用目录：将所有单链PDB文件放在一个目录中
2. 使用zip文件：将所有单链PDB文件打包成一个zip文件

### 示例命令：
```bash
# 使用目录方式
python3 main.py --mode=0 -F=example/6824.mrc -P=example -M=example/input_info.txt --config=config/diffmodeler.json --contour=2 --gpu=0 --resolution=5.8

# 使用zip文件方式
python3 main.py --mode=0 -F=example/6824.mrc -P=example/6824.zip -M=example/input_info.txt --config=config/diffmodeler.json --contour=2 --gpu=0 --resolution=5.8
```

### 2. 使用序列进行蛋白质复合物建模（EBI搜索）

如果您只有map和对应的序列，可以使用此模式。即使只知道部分序列也可以运行。
**注意：如果有超过4条非相同的链，请使用我们的[在线服务器](https://em.kiharalab.org/algorithm/DiffModeler(seq))**。当有很多非相同序列时，EBI的API响应会很慢。

命令格式：
```bash
python3 main.py --mode=1 -F=[Map_Path] -P=[fasta_path] --config=[pipeline_config_file] --contour=[Contour_Level] --gpu=[GPU_ID] --resolution=[resolution]
```

**重要：请在config/diffmodeler.json中更新"email"字段为您的邮箱。**

如果要使用基于域的结构建模，可以添加`--domain`选项。程序会首先调用SWORD2将每个单链分割成不同的域，然后使用这些域结构进行建模。或者，您可以使用[SWORD2服务器](https://www.dsimb.inserm.fr/SWORD2/index.html)探索不同的域分割选择，并将域结构作为单链结构提供给DiffModeler。

fasta文件示例：
```
>A,B,C,D
MATPAGRRASETERLLTPNPGYGTQVGTSPAPTTPTEEEDLRR
>E,F
VVTFREENTIAFRHLFLLGYSDGSDDTFAAYTQEQLYQ
```
ID行只能包含链ID，不能包含其他信息。如果多个链包含相同的序列，请使用逗号","分隔不同的链。
在这个例子中，总共有6条链，其中A,B,C,D共享相同的序列，E,F共享另一个相同的序列。

### 3. 使用序列进行蛋白质复合物建模（本地序列数据库）

如果您有超过4条非相同的链，可以设置本地序列数据库来运行DiffModeler。
**这是服务器使用的模式。我们强烈建议直接使用[在线服务器](https://em.kiharalab.org/algorithm/DiffModeler(seq))**

#### 3.1 安装Blast
请按照[NCBI网站](https://blast.ncbi.nlm.nih.gov/doc/blast-help/downloadblastdata.html)的说明在本地安装Blast。

#### 3.2 下载并安装数据库
从https://huggingface.co/datasets/zhtronics/BLAST_RCSB_AFDB/tree/main 下载处理好的数据库，并解压到`data`目录。

您也可以使用命令行：
```bash
wget https://huggingface.co/datasets/zhtronics/BLAST_RCSB_AFDB/resolve/main/data.tar.gz.aa
wget https://huggingface.co/datasets/zhtronics/BLAST_RCSB_AFDB/resolve/main/data.tar.gz.ab
wget https://huggingface.co/datasets/zhtronics/BLAST_RCSB_AFDB/resolve/main/data.tar.gz.ac
cat data.tar.gz.aa data.tar.gz.ab data.tar.gz.ac >data.tar.gz
tar -xzvf data.tar.gz
```

#### 3.3 运行DiffModeler
配置环境后，运行：
```bash
python3 main.py --mode=2 -F=[Map_Path] -P=[fasta_path] --config=[pipeline_config_file] --contour=[Contour_Level] --gpu=[GPU_ID] --resolution=[resolution]
```

### 结果可视化

运行脚本后，生成的cif文件将保存在`Predict_Result/[map_name]/DiffModeler.cif`中。每个单链的拟合得分保存在cif文件的occupency字段中。

要在PyMol中可视化拟合得分，可以在加载cif文件后运行：
```
spectrum q, red_white_blue, all, 0,1
```
这里蓝色表示拟合良好的链，红色表示可能拟合不好的链。

您也可以通过`--output`指定作业的输出目录。

## 训练自己的模型

### 1. 数据准备
数据应该放在一个目录[data_path]下，按照如下结构组织：
```
-[EMD-ID1]
  --input_1.npy
  --output_1.npy
  --input_2.npy
  --output_2.npy
  ...
-[EMD-ID2]
  --input_1.npy
  --output_1.npy
  ...
```
每个子目录对应不同的EM map数据。每个`input_[k].npy`和`output_[k].npy`是第k个样本的输入和目标。数据shape应该是[K,W,H]，其中K是通道数，W是box的宽度，H是box的高度。

用于训练和验证的map IDs [EMD-ID]应该准备在一个txt文件中，每行记录一个[EMD-ID]。

### 2. 训练配置
在`config/diffmodeler_train.json`中设置输入输出通道：
```json
"data": {
    "input_channel": [0],
    "output_channel":[0]
},
```

如果更改了通道数，需要相应更新网络配置：
```json
"unet": {
    "in_channel": 3,
    "out_channel": 1,
    ...
```

### 3. 开始训练
```bash
python3 train.py -F [data_path] --info_txt [info_txt_path] --config config/diffmodeler_train.json --gpu [gpu_id] --output [output_path]
```

## 评估模型结果

### 1. 转换.cif到.pdb
首先使用maxit将.cif格式转换为.pdb格式：[安装maxit](https://sw-tools.rcsb.org/apps/MAXIT/index.html)

然后运行：
```bash
maxit -input DiffModeler.cif -output DiffModeler.pdb -o 2
```
注意：对于超过9999个残基的大结构，请为每条链从1重新编号残基ID。

### 2. 使用MMalign评估
可以选择安装MMAlign或在线运行：[MMalign](https://zhanggroup.org/MM-align/)

运行以下命令比较DiffModeler.pdb和native.pdb：
```bash
./MMalign DiffModeler.pdb native.pdb >report.txt
```

评估指标在report.txt中：
- TM-score是第二个值，由第二个结构(native.pdb)归一化
- Align Ratio通过将报告的align长度除以原生结构的长度计算
- 序列一致性通过Seq_ID*Align-Ratio计算

## 示例

### 输入文件
- 冷冻电镜图（mrc格式）
- AlphaFold/模板单链结构和指示路径的信息文件
- 我们的示例输入可以在[这里](https://github.com/kiharalab/DiffModeler/tree/master/example)找到

### 输出文件
- DiffModeler.cif：记录最终建模的蛋白质复合物结构的CIF文件
- 我们的示例输出可以在[这里](https://kiharalab.org/emsuites/diffmodelder_example/output)找到。所有中间结果也保存在这里。

## 注意事项

较新版本的Intel MKL会导致pyTorch出现以下错误（由于符号被移除）：
```
ImportError undefined symbol: iJIT_NotifyEvent
```

我们已经在environment.yml和requirements.txt中将版本固定为较旧版本。任何之前的安装都应该正常工作。

如果您遇到这个问题，请先激活conda环境，然后运行：
```bash
conda install mkl==2024.0
