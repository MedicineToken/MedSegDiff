

# MedSegDiff: Medical Image Segmentation with Diffusion Model
MedSegDiff a Diffusion Probabilistic Model (DPM) based framework for Medical Image Segmentation. The algorithm is elaborated in our paper [MedSegDiff: Medical Image Segmentation with Diffusion Probabilistic Model](https://arxiv.org/abs/2211.00611).
## News
- 22-11-30. This project is still quickly updating. Check TODO list to see what will be released next.
- 22-12-03. BraTs2020 bugs fixed. Example case added.
- 22-12-15. Fix multi-gpu distributed training.
- 22-12-16. DPM-Solver ✖️ MedSegDiff DONE 🥳 Now [DPM-Solver](https://github.com/LuChengTHU/dpm-solver) is avaliable in MedsegDiff. Enjoy its lightning-fast sampling (1000 steps ❌ 20 steps ⭕️) by setting ``--dpm_solver True``. 
- 22-12-23. Fixed some bugs of DPM-Solver.
## Example Cases
### Melanoma Segmentation from Skin Images
1. Download ISIC dataset from https://challenge.isic-archive.com/data/. Your dataset folder under "data_dir" should be like:

ISIC/

     ISBI2016_ISIC_Part3B_Test_Data/...
     
     ISBI2016_ISIC_Part3B_Training_Data/...
     
     ISBI2016_ISIC_Part3B_Test_GroundTruth.csv
     
     ISBI2016_ISIC_Part3B_Training_GroundTruth.csv
    
2. For training, run: ``python scripts/segmentation_train.py --data_dir input data direction --out_dir output data direction --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 8``
    
3. For sampling, run: ``python scripts/segmentation_sample.py --data_dir input data direction --out_dir output data direction --model_path saved model --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --num_ensemble 5``


In default, the samples will be saved at `` ./results/`` 
### Brain Tumor Segmentation from MRI
1. Download BRATS2020 dataset from https://www.med.upenn.edu/cbica/brats2020/data.html. Your dataset folder should be like:
~~~
data
└───training
│   └───slice0001
│       │   t1.nii.gz
│       │   t2.nii.gz
│       │   flair.nii.gz
│       │   t1ce.nii.gz
│       │   seg.nii.gz
│   └───slice0002
│       │  ...
└───testing
│   └───slice1000
│       │   t1.nii.gz
│       │   t2.nii.gz
│       │   flair.nii.gz
│       │   t1ce.nii.gz
│   └───slice1001
│       │  ...
~~~
    
2. For training, run: ``python scripts/segmentation_train.py --data_dir (where you put data folder)/data/training --out_dir output data direction --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 8``

3. For sampling, run: ``python scripts/segmentation_sample.py --data_dir (where you put data folder)/data/testing --out_dir output data direction --model_path saved model --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --num_ensemble 5``


### Ohter Examples
...
### Run on  your own dataset
It is simple to run MedSegDiff on the other datasets. Just write another data loader file following `` ./guided_diffusion/isicloader.py`` or `` ./guided_diffusion/bratsloader.py``.  Welcome to open issues if you meet any problem. It would be appreciated if you could contribute your dataset extensions. Unlike natural images, medical images vary a lot depending on different tasks. Expanding the generalization of a method requires everyone's efforts.
## Suggestions for Hyperparameters and Training
To train a fine model, i.e., MedSegDiff-B in the paper, set the model hyperparameters as:
~~~
--image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 
~~~
diffusion hyperparameters as:
~~~
--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False
~~~
To speed up the sampling:
~~~
--diffusion_steps 50 --dpm_solver True 
~~~
run on multiple GPUs:
~~~
--multi-gpu 0,1,2 (for example)
~~~
training hyperparameters as:
~~~
--lr 5e-5 --batch_size 8
~~~
and set ``--num_ensemble 5`` in sampling.

Run about 100,000 steps in training will be converged on most of the datasets. Note that although loss will not decrease in most of the later steps, the quality of the results are still improving. Such a process is also observed on the other DPM applications, like image generation. Hope someone smart can tell me why🥲.

I will soon publish its performance under smaller batch size (suitable to run on 24GB GPU) for the need of comparison🤗.

A setting to unleash all its potential is (MedSegDiff++):
~~~
--image_size 256 --num_channels 512 --class_cond False --num_res_blocks 12 --num_heads 8 --learn_sigma True --use_scale_shift_norm True --attention_resolutions 24 
~~~
Then train it with batch size ``--batch_size 64`` and sample it with ensemble number ``--num_ensemble 25``.
## Be a part of MedSegDiff ! Authors are YOU !
Welcome to contribute to MedSegDiff. Any technique can improve the performance or speed up the algorithm is appreciated🙏. I am writting MedSegDiff V2, aiming at Nature journals/CVPR like publication. I'm glad to list the contributors as my co-authors🤗.
## TODO LIST

- [x] Fix bugs in BRATS. Add BRATS example.
- [ ] Release REFUGE and DDIT dataloaders and examples
- [x] Speed up sampling by DPM-solver
- [ ] Inference of depth
- [x] Fix bugs in Multi-GPU parallel
- [ ] Sample and Vis in training
- [ ] Release pre processing and post processing
- [ ] Release evaluation
- [ ] Deploy on HuggingFace
- [ ] yaml configuration

## Thanks
Code is copied a lot from [openai/improved-diffusion](https://github.com/openai/improved-diffusion), [WuJunde/ MrPrism](https://github.com/WuJunde/MrPrism), [WuJunde/ DiagnosisFirst](https://github.com/WuJunde/DiagnosisFirst), [LuChengTHU/dpm-solver](https://github.com/LuChengTHU/dpm-solver), [JuliaWolleb/Diffusion-based-Segmentation](https://github.com/JuliaWolleb/Diffusion-based-Segmentation), [hojonathanho/diffusion](https://github.com/hojonathanho/diffusion), [guided-diffusion](https://github.com/openai/guided-diffusion), [bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets](https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets), [nnUnet](https://github.com/MIC-DKFZ/nnUNet), [lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch)
## Cite
Please cite
~~~
@article{wu2022medsegdiff,
  title={MedSegDiff: Medical Image Segmentation with Diffusion Probabilistic Model},
  author={Wu, Junde and Fang, Huihui and Zhang, Yu and Yang, Yehui and Xu, Yanwu},
  journal={arXiv preprint arXiv:2211.00611},
  year={2022}
}
~~~

