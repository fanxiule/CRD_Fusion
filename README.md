# CRD-Fusion

This is the official repository for our paper
> Occlusion-Aware Self-Supervised Stereo Matching with Confidence Guided Raw Disparity Fusion
>
> by Xiule Fan, Soo Jeon, Baris Fidan
> Conference on Robots and Vision 2022 (Oral)

This branch contains some additional files for custom datasets collected by a RealSense and a Zed camera.

## Installation

The system has been tested with PyTorch 1.9, CUDA 11.1, Python 3.7 both on Ubuntu 18.04 and CentOS 7.9. You can set up
the environment easily by using [conda](https://www.anaconda.com/products/individual)
and run

```
conda env create -f environment.yml
```

## Pretrained Models

Each pretrained model is saved in a `.zip` folder. To be consistent with the rest of this document, create a directory
called `models`. Unzip the pretrained models and place them in `/models`. The file structure should be like

```
CRD_Fusion/
├── assets
├── data_preprocess
├── datasets
├── models
│   ├── KITTI2012
│   ├── KITTI2015
│   ├── SceneFlow
├── networks
├── .gitgnore
├── crd_fusion_net.py
...
```

| Scene Flow |  KITTI 2012 | KITTI 2015 | 
|---|---|---|
| [OneDrive](https://1drv.ms/u/s!Ai577MWqjhXlkAJgYk_IvF5xPKTs) |  [OneDrive](https://1drv.ms/u/s!Ai577MWqjhXlkARGJeKrgWFGRSbW) | [OneDrive](https://1drv.ms/u/s!Ai577MWqjhXlkAO8x_ffk5u7d-sX) | 

## Demo

A demo of our is provided by a jupyter notebook. Run the command below to launch the demo.

```
jupyter notebook example.ipynb
```

## Datasets

[Scene Flow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) (final pass),
[KITTI 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo),
and [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)
are used in this work. Assuming the root folder for the datasets is `~/Documents/Datasets`, move the files according to
the following structure after downloading the datasets.

```
~/Documents/Datasets
├── SceneFlow
│   ├── driving
│   │   ├── disparity
│   │   ├── frames_finalpass
│   ├── flyingThings3D
│   │   ├── disparity
│   │   ├── frames_finalpass
│   ├── monkaa
│   │   ├── disparity
│   │   ├── frames_finalpass
├── kitti2012
│   ├── training
│   │   ├── colored_0
│   │   ├── colored_1
│   │   ├── disp_noc
│   │   ├── disp_occ
│   ├── testing
│   │   ├── colored_0
│   │   ├── colored_1
├── kitti2015
│   ├── training
│   │   ├── image_2
│   │   ├── image_3
│   │   ├── disp_noc_0
│   │   ├── disp_occ_0
│   ├── testing
│   │   ├── image_2
│   │   ├── image_3
├── realsense
│   ├── left
│   │   ├── 0.png
│   │   ├── ...
│   ├── right
│   │   ├── 0.png
│   │   ├── ...
└── zed
    ├── left
    │   ├── 0.png
    │   ├── ...
    └── right
        ├── 0.png
        └── ...
```

## Data Preprocessing

To compute raw disparity and confidence maps for the public datasets, run the following

```
python preprocess_data.py --dataset_path ~/Documents/Datasets/ \
                          --dataset_name SceneFlow \
                          --max_disp 192 \
                          --block_size 3 \
                          --match_method SGBM \
                          --device cuda \
                          --full_ZSAD \
                          --train_val_split_per 0.8 \
                          --random_seed 75
```

You can choose `--dataset_name` from `SceneFlow`, `kitti2012`, `kitti2015`, `realsense`, and `zed`. To separate the
RealSense/Zed dataset or the training set of KITTI 2012/2015 into training and validation splits,
specify `--train_val_split_per` and `--random_seed`.

## Training

Train the model on Scene Flow by

```
python train.py --data_path ~/Documents/Datasets/ --log_dir models \
                --model_name train_SceneFlow \
                --dataset SceneFlow --resized_height 256 --resized_width 512 \
                --downscale 1 --max_disp 192 --batch_size 8 \
                --learning_rate 0.001 --num_epochs 15 --scheduler_step 10 --lr_change_rate 0.1 \
                --conf_threshold 0.8 --imagenet_norm --feature_downscale 3 --multi_step_upsample \
                --fusion \
                --loss_conf \
                --occ_detection \
                --supervision_weight 0.7 \
                --photo_weight 3 \
                --smooth_weight 0.45 \
                --occ_weight 0.75 \
                --occ_epoch -1 --device cuda --num_workers 2 \
                --early_log_frequency 200 --late_log_frequency 2000 --early_late_split 4000 --save_frequency 5
```

Checkpoints and TensorBoard events are saved in `models/train_SceneFlow` or other directory specified by `--log_dir`
and `--model_name`. You can use TensorBoard to visualize the intermediate results.

To fine tune the model on KITTI 2012/2015, run

```
python train.py --data_path ~/Documents/Datasets/ --log_dir models \
                --model_name train_KITTI2015 \
                --dataset kitti2015_full --resized_height 256 --resized_width 512 \
                --downscale 1 --max_disp 192 --batch_size 8 \
                --learning_rate 0.0001 --num_epochs 1000 --scheduler_step 200 --lr_change_rate 0.5 \
                --conf_threshold 0.8 --imagenet_norm --feature_downscale 3 --multi_step_upsample \
                --fusion \
                --loss_conf \
                --occ_detection \
                --supervision_weight 8.5 \
                --photo_weight 0.8 \
                --smooth_weight 0.05 \
                --occ_weight 0.3 \
                --occ_epoch -1 --device cuda --num_workers 2 \
                --pretrained_model_path models/SceneFlow \
                --early_log_frequency 20 --late_log_frequency 200 --early_late_split 1000 --save_frequency 200
```

This command assumes you have downloaded the model pretrained on Scene Flow or trained your own model. The weights
should be saved in `models/SceneFlow` or other directory specified by `--pretrained_model_path`. In the
`models/SceneFlow` folder, there should be `adam.pth`, `disp_est.pth`, `disp_refine.pth`, and `extractor.pth`.
For `--dataset`, You can choose from `kitti2015`, `kitti2015_full`, `kitti2012`, `kitti2012_full`, `realsense`,
and `zed`. Choosing the dataset with `_full` means all training images will be used to fine tune the models. For
datasets without `_full`, only the training split of the training images is used for fine-tuning. You may need to tune
some hyperparameters (e.g., weights in the training loss, number of epochs, scheduler step, etc.) to fine tune on your
custom datasets.

## Evaluation

Use the command below to evaluate the model on Scene Flow

```
python eval.py --data_path ~/Documents/Datasets/ --checkpt models/SceneFlow --log_dir models \
               --dataset SceneFlow --resized_height 544 --resized_width 960 --downscale 1 --max_disp 192 \
               --model_name eval_SceneFlow --device cuda --num_workers 0 --log_frequency 100 \
               --conf_threshold 0.8 --imagenet_norm --feature_downscale 3 --multi_step_upsample \
               --fusion \
               --occ_detection
```

It is assumed that the pretrained weights are saved in `models/SceneFlow`. The TensorBoard event is saved
in `models/eval_SceneFlow` or other directory specified by `--log_dir` and `--model_name`.

To perform evaluation on the validation split of KITTI 2012/2015 datasets, change `--dataset` to `kitti2012`
or `kitti2015`, change `--checkpt` to `models/KITTI2012` or `models/KITTI2015`. Lastly, set `--resized_height` to 376
and `--resized_width` to 1248. If you want to save the intermediate results as a TensorBoard event,
set `--log_frequency` to 5. You can also change `--dataset` to `realsense` or `zed` to evaluate on the custom
datasets. `--resized_height`, `--resized_width`, and `--checkpt` need to be updated accordingly as well.

<strong>Note</strong>: The provided pretrained models for KITTI 2012/2015 have been trained using `kitti2012_full`
or `kitti2015_full`. Setting `--dataset` to `kitti2012` or `kitti2015` will evaluate the model on the validation split
of the training images, which will lead to biased results.

## Prediction

Run the following command to make predictions on KITTI 2012/2015 test set. After executing the command below, the frame
rate of the pipeline will be printed out. The frame rate includes both the confidence generation step and forward pass
of our CRD-Fusion network.

```
python predict_kitti.py --data_path ~/Documents/Datasets/ \
                        --checkpt models/KITTI2015 \
                        --log_dir models \
                        --model_name test_kitti2015 \
                        --dataset kitti2015_test \
                        --save_pred
```

You can replace `kitti2015_test`with `kitti2012_test` for KITTI 2012. By setting the `--save_pred` flag, the predictions
are saved in `models/test_kitti2015` or a directory specified by `--log_dir` and `--model_name`. The predicted disparity
maps are saved as 16-bit `.png` files, while confidence maps and occlusion masks are saved in `.npy` format.

<strong>Note</strong>: If you ran the command shown in the <strong>Data Preprocessing</strong> session for KITTI
2012/2015, the raw disparity maps and confidence maps have already been generated for the test sets.`predict_kitti.py`
actually performs the confidence generation step again so that it is also considered in the runtime.

## Citation

If you find our work useful for your research, please consider citing our paper.

```
@inproceedings{crd_fusion,
  author = {Fan, Xiule and Jeon, Soo and Fidan, Baris},
  title = {Occlusion-Aware Self-Supervised Stereo Matching with Confidence Guided Raw Disparity Fusion},
  booktitle = {Conference on Robots and Vision},
  year = {2022}
}
```

## Acknowledgment

Some of the code is inspired by [MaskFlowNet](https://github.com/microsoft/MaskFlownet) and StereoNet implemented in an
earlier version of this [repository](https://github.com/meteorshowers/X-StereoLab). We would like to thank the original
authors for their amazing work. 