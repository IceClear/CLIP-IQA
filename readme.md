
## Exploring CLIP for Assessing the Look and Feel of Images (AAAI 2023)

[Paper](https://arxiv.org/abs/2207.12396)


![visitors](https://visitor-badge.laobi.icu/badge?page_id=IceClear/CLIP-IQA)


[Jianyi Wang](https://iceclear.github.io/), [Kelvin C.K. Chan](https://ckkelvinchan.github.io/), [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/)

S-Lab, Nanyang Technological University

<img src="https://user-images.githubusercontent.com/22350795/202890659-2b73008f-fc0d-49c6-8bc8-0c1f07df5d36.png" width="800px"/>

### TODO
- [ ] Website release
- [x] ~~Code release~~

### Dependencies and Installation
The same as MMEditing(https://mmediting.readthedocs.io/en/latest/install.html), currently is using an old version.
```
conda create -n clipiqa python=3.6
pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html # For CUDA 10.1
pip install -r requirements.txt
pip install -v -e .
```

### Running Examples

#### Test CLIP-IQA on KonIQ-10k

```
python demo/clipiqa_koniq_demo.py
```

#### Test CLIP-IQA on Live-iWT

```
python demo/clipiqa_liveiwt_demo.py
```

#### Train CLIP-IQA+ on KonIQ-10k

```
# Support dist training as MMEditing
python tools/train.py configs/clipiqa/clipiqa_coop_koniq.py
```

#### Test CLIP-IQA+ on KonIQ-10k

```
python demo/clipiqa_koniq_demo.py --config configs/clipiqa/clipiqa_coop_koniq.py --checkpoint ./iter_80000.pth
```

[Note] You may change prompts for different datasets, please refer to [config files](https://github.com/IceClear/CLIP-IQA/tree/main/configs/restorers/clipiqa/clipiqa_coop_koniq.py#L9) for details.

[Note] For testing on a single image, please refer to [here](https://github.com/IceClear/CLIP-IQA/tree/main/demo/clipiqa_single_image_demo.py) for details.

### Demo

#### :sparkles: Versatile Quality Assessment
<img src="https://user-images.githubusercontent.com/22350795/202886677-63c6af8d-4ae8-4c88-a6e6-b0b980738634.png" width="800px"/>

#### :sparkles: Demo for IQA on SPAQ
<img src="assets/SPAQ-exp.png" width="800px"/>

#### :sparkles: Demo for Abstract Perception on AVA
<img src="assets/AVA-exp.png" width="800px"/>

For more evaluation, please refer to our [paper](https://arxiv.org/abs/2207.12396) for details.

### Citation
If our work is useful for your research, please consider citing:

    @inproceedings{wang2022exploring,
        author = {Wang, Jianyi and Chan, Kelvin CK and Loy, Chen Change},
        title = {Exploring CLIP for Assessing the Look and Feel of Images},
        booktitle = {AAAI},
        year = {2023}
    }

### License

This project is licensed under <a rel="license" href="https://github.com/IceClear/CLIP-IQA/blob/master/LICENSE">NTU S-Lab License 1.0</a>. Redistribution and use should follow this license.

### Acknowledgement

This project is based on [MMEditing](https://github.com/open-mmlab/mmediting) and [CLIP](https://github.com/openai/CLIP). Thanks for their awesome works.

### Contact
If you have any question, please feel free to reach me out at `iceclearwjy@gmail.com`.
