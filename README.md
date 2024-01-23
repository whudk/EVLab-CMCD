# EVLab-SHPCD
Cross-Modal Change Detection Using Historical Land Use Base Maps and Current Remote Sensing Images
EVLab-SHPCD dataset is a cross-modal change detection benchmark proposed and leaded by Prof. Xiangyun Hu and created by Dr. Kai Deng and other EVLab members.
EVLab being the abbreviation for the Earth vision Laboratory at Wuhan University in Hubei Province, China, and SHPCD being the name of the dataset.

The dataset is divided into a training set containing 5225 pairs and a validation set with 397 pairs. Each image has a pixel resolution of 512×512 and a spatial resolution of 0.8 meters.

Image Components
  Hlubm (Historical Land Use Base Map):
    -Grayscale map depicting historical land use classifications.
    -Pixel values range from 0 to 85, with each value corresponding to a specific land use classification category.
  Crism (Current Remote Sensing Image):
    -Represents the contemporary remote sensing image.
  Target:
    -Manually annotated binary changes.
    -0 indicates the background.
    -1 indicates newly developed land, including buildings, roads, and fill areas.
![](https://github.com/whudk/EVLab-SHPCD/blob/main/images/evlab_shpcd.png)
