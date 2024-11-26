# Data Converter for Nuplan Dataset

## Overview

This repo is to convert raw nuplan data to the format needed by `VAD` to train end-2-end autonomous driving algorithms.

Main contribution includes:

- convert raw `.db` to `.pkl`, see [vad_nuplan_converter.py](scripts/vad_nuplan_converter.py) for details

- convert raw `.gpkg` to `.json`, including necessary layer mapping. see [gpkg2json.py](scripts/gpkg2json.py) for details

- Visualizer to verify the format of converted data. see [viz_tools](viz_tools)

## Env Setup

`python 3.10` is recommended

```shell
python3 -m pip install -e . && pip install -r requirements.txt
```

```shell
cd nuplan-devkit && python3 -m pip install -e . && pip install -r requirements.txt
```
