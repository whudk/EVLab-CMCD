# 📦 EVLab-SHPCD: A Cross-Modal Change Detection Dataset

**EVLab-SHPCD** is a cross-modal change detection benchmark proposed and led by *Prof. Xiangyun Hu*, and constructed by *Dr. Kai Deng* and other members of the **Earth Vision Laboratory (EVLab)** at Wuhan University, Hubei Province, China. The dataset is designed to promote research on detecting land use changes by combining **historical land use base maps** with **current high-resolution remote sensing imagery**.

## 🗂️ Dataset Overview

- **Training Set**: 5,225 image pairs  
- **Validation Set**: 397 image pairs  
- **Image Size**: 512 × 512 pixels  
- **Spatial Resolution**: 0.8 meters  

Each image pair consists of the following three components:

1. **Hlubm** (Historical Land Use Base Map)  
   - A grayscale image representing historical land use categories  
   - Pixel values range from **0 to 85**, each corresponding to a specific land use type  

2. **Crism** (Current Remote Sensing Image)  
   - A contemporary high-resolution optical remote sensing image  

3. **Target** (Change Annotation)  
   - A binary label map manually annotated  
   - Pixel value `0`: background / no change  
   - Pixel value `255`: newly developed land, including buildings, roads, and filled areas  

<p align="center">
  <img src="https://github.com/whudk/EVLab-SHPCD/blob/main/images/evlab_shpcd.png" width="600"/>
</p>

---

## 📥 Download

- **Baidu Cloud**: [https://pan.baidu.com/s/1Bq_P7wr5z6d3UixCyTtPtA](https://pan.baidu.com/s/1Bq_P7wr5z6d3UixCyTtPtA)  
- **Access Code**: `cmcd`  

> **Note**: This dataset is released strictly for **academic research purposes only**. Any **commercial use is prohibited**.

---

## 📖 Citation

If you use EVLab-SHPCD in your work, please cite the following paper:

```bibtex
@article{DENG2024114,
  title     = {Cross-modal change detection using historical land use maps and current remote sensing images},
  journal   = {ISPRS Journal of Photogrammetry and Remote Sensing},
  volume    = {218},
  pages     = {114-132},
  year      = {2024},
  issn      = {0924-2716},
  doi       = {https://doi.org/10.1016/j.isprsjprs.2024.10.010},
  url       = {https://www.sciencedirect.com/science/article/pii/S0924271624003873},
}
