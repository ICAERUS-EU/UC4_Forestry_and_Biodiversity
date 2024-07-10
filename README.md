# UC4_Forestry_and_Biodiversity
## Table Of Contents
- [Summary](#summary)
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [Development](#development)
- [Testing](#testing)
- [Documentation](#documentation)
- [License](#license)
- [Support](#support)
- [Security](#security)
- [Acknowledgments](#acknowledgments)
- [Resources](#resources)
- [Gallery](#gallery)
- [Deployment](#deployment)
- [Demo](#demo)
- [Dependencies](#dependencies)
- [Known Issues](#known-issues)
- [Roadmap](#roadmap)


## Introduction
Two python libraries are added:
hyperspectral-lib - library for hyperspectral data processing (specifically Specim hyperspectral camera used)
sentinel-tools-lib - library for sentinel data colelction and processing using sentinel-hub python API. Only AWS account is required, and not sentinelhub account, all data is downloaded from Sentinel-2 S3 inventory.
boar-utils  - library for boar thermal data processing base and other utilities. 
boar-detector - models for boar entity detection in thermal images and counting using the drone coordinates. Algorithm detects the unique boar entities in images and removes duplicates when drone flies over the same area.
boars-yolo  -  wrapper for yolov5 model for boar detection. Using the objects from boar-utils and sending the detected boars to boar-detector model.

## Instalation
Libraries can be installed in a Pipenv or Virtualenv using `pip install -e /path/to/libarary/folder/with/setup.py`
Install library dependencies from library Pipfile or requirements. 


## Acknowledgements
This project is funded by the European Union, grant ID 101060643.

<img src="https://rea.ec.europa.eu/sites/default/files/styles/oe_theme_medium_no_crop/public/2021-04/EN-Funded%20by%20the%20EU-POS.jpg" alt="https://cordis.europa.eu/project/id/101060643" width="200"/>
