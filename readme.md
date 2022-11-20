
## Exploring CLIP for Assessing the Look and Feel of Images (AAAI 2023)

[Paper](https://arxiv.org/abs/2207.12396)


![visitors](https://visitor-badge.laobi.icu/badge?page_id=IceClear/CLIP-IQA)


[Jianyi Wang](https://iceclear.github.io/), [Kelvin C.K. Chan](https://ckkelvinchan.github.io/), [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/)

S-Lab, Nanyang Technological University

<img src="https://user-images.githubusercontent.com/22350795/202890659-2b73008f-fc0d-49c6-8bc8-0c1f07df5d36.png" width="800px"/>

<!-- **[<font color=#d1585d>News</font>]**: :whale: *Due to copyright issues, we have to delay the release of the training code (expected by the end of this year). Please star and stay tuned for our future updates!*
### Update
- **2022.10.05**: Support video input `--input_path [YOUR_VIDOE.mp4]`. Try it to enhance your videos! :clapper:
- **2022.09.14**: Integrated to :hugs: [Hugging Face](https://huggingface.co/spaces). Try out online demo! [![Hugging Face](https://img.shields.io/badge/Demo-%F0%9F%A4%97%20Hugging%20Face-blue)](https://huggingface.co/spaces/sczhou/CodeFormer)
- **2022.09.09**: Integrated to :rocket: [Replicate](https://replicate.com/explore). Try out online demo! [![Replicate](https://img.shields.io/badge/Demo-%F0%9F%9A%80%20Replicate-blue)](https://replicate.com/sczhou/codeformer)
- **2022.09.04**: Add face upsampling `--face_upsample` for high-resolution AI-created face enhancement.
- **2022.08.23**: Some modifications on face detection and fusion for better AI-created face enhancement.
- **2022.08.07**: Integrate [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) to support background image enhancement.
- **2022.07.29**: Integrate new face detectors of `['RetinaFace'(default), 'YOLOv5']`.
- **2022.07.17**: Add Colab demo of CodeFormer. <a href="https://colab.research.google.com/drive/1m52PNveE4PBhYrecj34cnpEeiHcC5LTb?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>
- **2022.07.16**: Release inference code for face restoration. :blush:
- **2022.06.21**: This repo is created. -->

### TODO
- [ ] Code release
<!-- - [ ] Add checkpoint for face colorization
- [ ] Add training code and config files
- [x] ~~Add background image enhancement~~ -->

#### :sparkles: Versatile Quality Assessment
<img src="https://user-images.githubusercontent.com/22350795/202886677-63c6af8d-4ae8-4c88-a6e6-b0b980738634.png" width="800px"/>

#### :sparkles: Demo for IQA on SPAQ
<img src="assets/SPAQ-exp.png" width="800px"/>

#### :sparkles: Demo for Abstract Perception on AVA
<img src="assets/AVA-exp.png" width="800px"/>

For more evaluation, please refer to our [Paper](https://arxiv.org/abs/2207.12396) for details.

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
