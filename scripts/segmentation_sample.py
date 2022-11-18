"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import nibabel as nib
from visdom import Visdom
viz = Visdom(port=8850)
import sys
import random
sys.path.append(".")
import numpy as np
import time
import torch as th
from PIL import Image
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.isicloader import ISICDataset
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import torchvision.transforms as transforms
from torchsummary import summary
seed=10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def dice_score(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()

def tensor_to_img_array(tensor):
    image = tensor.cpu().detach().numpy()
    image = np.transpose(image, [0, 2, 3, 1])
    return image

def export(tensor, img_path=None):
    # image_name = image_name or "image.jpg"
    c = tensor.size(1)
    if c == 3:
        vutils.save_image(tensor, fp = img_path)
    else:
        w_map = tensor[:,-1,:,:].unsqueeze(1)
        w_map = 1 - tensor_to_img_array(w_map).squeeze()
        w_map = (w_map * 255).astype(np.uint8)

        Image.fromarray(w_map,'L').save(img_path)


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist(args)
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    transform_test = transforms.Compose([
    transforms.Resize((args.image_size,args.image_size)),
    # transforms.RandomCrop((img_size, img_size)),  # padding=10
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
    transforms.ToTensor(),
    ])

    ds = ISICDataset(args, args.data_dir, transform_test, transform_test, mode = 'Test')
    datal = th.utils.data.DataLoader(
        ds,
        batch_size=1,
        shuffle=False)
    data = iter(datal)
    all_images = []

    summary(model.to(dist_util.dev()), [(4, 256, 256),(1,)])

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    while len(all_images) * args.batch_size < args.num_samples:
        b, m, path = next(data)  #should return an image from the dataloader "data"
        c = th.randn_like(b[:, :1, ...])
        img = th.cat((b, c), dim=1)     #add a noise channel$
        print('img shape is',img.size())
        # img = b
        slice_ID=path[0].split("_")[-1].split('.')[0]

        # viz.image(visualize(img[0,0,...]), opts=dict(caption="img input0"))
        # viz.image(visualize(img[0, 1, ...]), opts=dict(caption="img input1"))
        # viz.image(visualize(img[0, 2, ...]), opts=dict(caption="img input2"))
        # viz.image(visualize(img[0, 3, ...]), opts=dict(caption="img input3"))
        # viz.image(visualize(img[0, 4, ...]), opts=dict(caption="img input4"))

        logger.log("sampling...")

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)


        for i in range(args.num_ensemble):  #this is for the generation of an ensemble of 5 masks.
            model_kwargs = {}
            start.record()
            sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )
            sample, x_noisy, org = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size), img,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

            end.record()
            th.cuda.synchronize()
            print('time for 1 sample', start.elapsed_time(end))  #time measurement for the generation of 1 sample

            s = th.tensor(sample)
            print('sample size is', s.size())
            # viz.image(visualize(sample[0, 0, ...]), opts=dict(caption="sampled output"))
            export(s, './results/'+str(slice_ID)+'_output'+str(i)+".jpg")
            # th.save(s, './results/'+str(slice_ID)+'_output'+str(i)) #save the generated mask

def create_argparser():
    defaults = dict(
        data_dir="./data/testing",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="",
        num_ensemble=1,      #number of samples in the ensemble
        gpu_dev = "0"
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":

    main()
