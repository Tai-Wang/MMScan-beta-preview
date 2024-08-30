# MMScan VG benchmark

## 🏠 About

This codebase preliminarily integrates the first 2 models shown below and will include the other two soon.

1. EmbodiedScan
2. ScanRefer
3. BUTD-DETR
4. ViL3DRef



## 📚 Basic Guide

This codebase is a beta version for understanding how we organize the MMScan's data and conduct the training and benchmark process. Next, we provide the guide for different aspects.

### Dataset

#### Annotation Files Obtaining
Follow the EmbodiedScan dataset instructions to download the annotation files. The annotation files are in the following format:

```
data
├── scannet
│   ├── scans
│   │   ├── <scene_id>
│   │   ├── ...
│   ├── posed_images
│   │   ├── <scene_id>
│   │   |   ├── *.jpg
│   │   |   ├── *.png
│   │   ├── ...
├── 3rscan
│   ├── <scene_id>
│   │   ├── sequence
│   │   |   ├── *.color.jpg
│   │   |   ├── *.depth.pgm
│   ├── ...
├── matterport3d
│   ├── <scene_id>
│   ├── ...
├── embodiedscan_occupancy
├── embodiedscan_infos_train_full.pkl
├── embodiedscan_infos_val_full.pkl
├── embodiedscan_infos_train_full_vg.json
├── embodiedscan_infos_val_full_vg.json
├── embodiedscan_infos_train_mini_vg.json
├── embodiedscan_infos_val_mini_vg.json
```

#### Base Data Preparation
Under `./base_data/` , run this code to obtain the 3d point cloud data adapted for the 3D Visual Grounding models:

``` bash
    python make_es_pcds.py
```

This will result in point clouds in the following format:
pcd_with_global_alignment/{scene_identifier}.pth, and the data can be accessed using the following code:

``` python
    pc, color, object_type_ids, instance_ids = torch.load(pcd_path)
    # pc: (N, 3) float, point cloud
    # color: (N, 3) float, point cloud color, range in [0, 1]
    # object_type_ids: (N,) int, object type id for each point
    # instance_ids: (N,) int, instance id for each point
```


### Models

The code of EmbodiedScan and ScanRefer are in separate folders under `./benchmark/`

#### EmbodiedScan

1. Please follow the guide in `./benchmark/EmbodiedScan/README.md` to install the EmbodiedScan environment.

2. We modify the code of EmbodiedScan to match MMScan data, below are the primary modifications we have made:

    * We add config files for training and testing on MMScan, under `./benchmark/EmbodiedScan/configs/grounding`, 
    * We add dataloader `mv_3dvg_dataset.py`,`pcd_3dvg_dataset.py` under `./benchmark/EmbodiedScan/embodiedScan/datasets` for both multi-view and 3d-point-cloud inputs.
    * We modify the model structure and loss function under `./benchmark/EmbodiedScan/embodiedScan/models`.
    * We add 3D Visual Grounding Metric adapted for MMScan under `./benchmark/EmbodiedScan/embodiedscan/eval/metrics/`


3. Train EmbodiedScan on MMScan data.

    ```bash
    cd ./EmbodiedScan
    python tools/train.py configs/grounding/pcd_4xb24-mmscan-20-100-vg-9dof.py --work-dir exps/Embodiedscan
    ```
#### ScanRefer

1. We modify the code of ScanRefer to match MMScan data, below are the primary modifications we have made:

    * We modify the dataloader to match the MMScan data format, under `./benchmark/ScanRefer/lib/dataset.py`

    * We add 3D Visual Grounding Metric adapted for MMScan under `./benchmark/ScanRefer/lib/grounding_metric.py`

    * We modify the model structure to match the MMScan data under `./benchmark/ScanRefer/models/`

2. Train scanrefer on MMScan data.

    ```bash
    cd ./ScanRefer
    bash train.sh
    ```

Note: In order to run the program successfully, you need to change the `/path/to/something` paths in the code to the corresponding data path.

   