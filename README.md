
<div align="center">
  <p>
    <a href="https://icaerus.eu" target="_blank">
      <img width="50%" src="https://icaerus.eu/wp-content/uploads/2022/09/ICAERUS-logo-white.svg"></a>
    <h3 align="center">UC4: Forestry and Biodiversity</h3>
    
   <p align="center">
    This repository contains Forestry and Biodiversity monitoring and analysis algorithms created for Sentinel-2 satellite and UAV mounted hyperspectral camera data 
    <br/>
    <br/>
    <br/>
    <br/>
    <a href="https://github.com/icaerus-eu/icaerus-repo-template/issues">Report Bug</a>
    -
    <a href="https://github.com/icaerus-eu/icaerus-repo-template/issues">Request Feature</a>
  </p>
</p>
</div>

![Downloads](https://img.shields.io/github/downloads/icaerus-eu/UC1_Crop_Monitoring/total) ![Contributors](https://img.shields.io/github/contributors/icaerus-eu/UC1_Crop_Monitoring?color=dark-green) ![Forks](https://img.shields.io/github/forks/icaerus-eu/UC1_Crop_Monitoring?style=social) ![Stargazers](https://img.shields.io/github/stars/icaerus-eu/UC1_Crop_Monitoring?style=social) ![Issues](https://img.shields.io/github/issues/icaerus-eu/UC1_Crop_Monitoring) ![License](https://img.shields.io/github/license/icaerus-eu/UC1_Crop_Monitoring) 

## Table Of Contents

* [Summary](#summary)
* [Structure](#structure)
* [Models](#models)
* [Authors](#authors)
* [Acknowledgements](#acknowledgements)

## Summary
Within this repository, you'll discover various models and computational tools designed for forestry and biodiversity. Models created for tree health monitoring system from Sentinel-2 multispectral data, tree health and fire monitoring using UAV mounted hyperspectral cameras, and automatic wild boar monitoring and detection from thermal cameras.

## Structure
The repository folders are structured as follow: 

- **data:** here you should add the Boar, Hyperspectral and Sentinel-2 datasets that you download from Zenodo.
- **models:** models developed for forestry and biodiversity monitoring
- **libs:** libraries created for hyperspectral and sentiinel-2 data processing
  - **hyperspectral-lib:** Python3 library created for hyperspectral data analysis, reading, writing and modelling purposes. 
  - **sentinel-tools-lib:** Python3 library created for sentinel data processing and analysis, with the capabilities of downloading sentinel data from AWS with the help of sentinelhub-py library. 
- **platform.json:** Structured information about the models and their parameters.
- **README.md:** This file, providing an overview of the repository.

## Models

The [models](https://github.com/ICAERUS-EU/UC4_Forestry_and_Biodiversity/tree/main/models) developed are the following:

#### _[Wild boar monitoring and detection model created using YOLOv5](https://github.com/ICAERUS-EU/UC4_Forestry_and_Biodiversity/tree/main/models/01_boar_detector_v1)_
This model was created using YOLOv5 object detection framework with and extension algorithms created to count unique wild boars in gathered UAV thermal data.


#### _[Sentinel-2 data preprocessing algorithms and pipeline](https://github.com/ICAERUS-EU/UC4_Forestry_and_Biodiversity/tree/main/models/02_Sentinel2_data_processing)_
Algorithms created for automatic raw Sentinel-2 multispectral data processing to a usable dataset based on regions of interest.

#### _[Automated region of interest monitoring models based on Sentinel-2 data](https://github.com/ICAERUS-EU/UC4_Forestry_and_Biodiversity/tree/main/models/03_Sentinel2_ROI_monitoring)_
Models created to automatically monitor regions of interest using Sentinel-2 data and extends the Sentinel-2 preprocessing algorithms.

#### _[Drone data processing pipelines and algorithms](https://github.com/ICAERUS-EU/UC4_Forestry_and_Biodiversity/tree/main/models/04_drone_data_processing)_
Algorithms created to process raw hyperspectral and thermal data gathered by the UAVs into datasets usable in modelling.


#### _[Forest health monitoring model using Sentinel-2 multispectral data](https://github.com/ICAERUS-EU/UC4_Forestry_and_Biodiversity/tree/main/models/05_Sentinel2_forest_health)_
Model created for monitoring forest health using processed Sentinel-2 multispectral and additional data.

#### _[Hyperspectral data processing algorithms](https://github.com/ICAERUS-EU/UC4_Forestry_and_Biodiversity/tree/main/models/06_hyperspectral_processing)_ and _[Hyperspectral calibration plate detection model](https://github.com/ICAERUS-EU/UC4_Forestry_and_Biodiversity/tree/main/models/07_hyperspectral_calibration_plate_detection)_
Base hyperspectral data processing algorithms and model created to automatically detect calibration plates in hyperspectral mission that are used to calibrate hyperspectral data from radiance values to reflectance.


#### _[Automated hyperspectral data clustering model with Convolutional Autoencoders](https://github.com/ICAERUS-EU/UC4_Forestry_and_Biodiversity/tree/main/models/08_hyperspectral_clustering_CAE)_
Model created to automatically find data clusters (data classes) in hyperspectral data cubes for further analysis base on Convolutional Autoencoders.


#### _[Forest health monitoring model using hyperspectral data](https://github.com/ICAERUS-EU/UC4_Forestry_and_Biodiversity/tree/main/models/09_hyperspectral_forest_health)_
Model for forest health monitoring on processend and clustered hyperspectral data.


#### _[Forest fuel evaluation model using hyperspectral data](https://github.com/ICAERUS-EU/UC4_Forestry_and_Biodiversity/tree/main/models/10_hyperspectral_forest_fuel)_
Model for forest fuel evaluation on processend and clustered hyperspectral data.


## Authors

Vytautas Paura - ART21 - [Vytautas Paura](https://github.com/VytautasPau) 

## Acknowledgements
This project is funded by the European Union, grant ID 101060643.

<img src="https://rea.ec.europa.eu/sites/default/files/styles/oe_theme_medium_no_crop/public/2021-04/EN-Funded%20by%20the%20EU-POS.jpg" alt="https://cordis.europa.eu/project/id/101060643" width="200"/>


