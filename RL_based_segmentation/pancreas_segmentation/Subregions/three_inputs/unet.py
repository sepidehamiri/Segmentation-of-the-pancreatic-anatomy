import os
import shutil
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    KeepLargestConnectedComponent,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd, FillHoles,
    RandRotate90d, AddChanneld,
    EnsureTyped, ConcatItemsd
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import UNet

from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    set_track_meta,
)


import nibabel as nib
import torch

print_config()

directory = "model_log/unet"
root_dir = tempfile.mkdtemp() if directory is None else directory
print("root dir", root_dir)

num_samples = 4

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

keys = ["image_pt", "image_ct", "image_sg"]
train_transforms = Compose(
    [
        LoadImaged(keys=keys, ensure_channel_first=True),
        # AddChanneld(keys=keys),

        # ScaleIntensityRanged(
        #     keys=["image_ct"], a_min=-100, a_max=250,
        #     b_min=0.0, b_max=1.0, clip=False,
        # ),
        # ScaleIntensityRanged(
        #     keys=["image_pt"], a_min=0, a_max=15,
        #     b_min=0.0, b_max=1.0, clip=False,
        # ),
        # CropForegroundd(keys=keys, source_key="image_ct"),
        Orientationd(keys=keys, axcodes="LAS"),
        EnsureTyped(keys=keys, device=device),
        RandCropByPosNegLabeld(
            keys=keys,
            label_key="image_sg",
            spatial_size=(64, 64, 64),
            pos=0,
            neg=1,
            num_samples=num_samples,
            image_key="image_ct",
            image_threshold=0,
        ),
        ConcatItemsd(keys=["image_pt", "image_ct"], name="image_petct", dim=0),

        RandFlipd(
            keys=["image_petct", "image_sg"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image_petct", "image_sg"],
            spatial_axis=[1],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image_petct", "image_sg"],
            spatial_axis=[2],
            prob=0.10,
        ),
        RandRotate90d(
            keys=["image_petct", "image_sg"],
            prob=0.10,
            max_k=3,
        ),
        RandShiftIntensityd(
            keys=["image_petct"],
            offsets=0.10,
            prob=0.50,
        ),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=keys, ensure_channel_first=True),
        # AddChanneld(keys=keys),

        # ScaleIntensityRanged(
        #     keys=["image_ct"], a_min=-100, a_max=250,
        #     b_min=0.0, b_max=1.0, clip=False,
        # ),
        # ScaleIntensityRanged(
        #     keys=["image_pt"], a_min=0, a_max=15,
        #     b_min=0.0, b_max=1.0, clip=False,
        # ),
        # CropForegroundd(keys=keys, source_key="image_ct"),
        Orientationd(keys=keys, axcodes="LAS"),
        ConcatItemsd(keys=["image_pt", "image_ct"], name="image_petct", dim=0),

        EnsureTyped(keys=["image_petct", "image_sg"], device=device),
    ]
)

data_dir = '/home/navid/Desktop/Papers/paper2/pancreas_segmentaion/Subregions/subregion_dataset'
images_sg = sorted(glob.glob(os.path.join(data_dir, '*', "mask*")))
images_pt = sorted(glob.glob(os.path.join(data_dir, '*', "registration.*")))
images_ct = sorted(glob.glob(os.path.join(data_dir, '*', "image*")))

data_dict = [
    {"image_pt": image_name_pt, "image_ct": image_name_ct, "image_sg": image_name_sg}
    for image_name_pt, image_name_ct, image_name_sg in zip(images_pt, images_ct, images_sg)
]
keys = ["image_pt", "image_ct", "image_sg"]


f1 = data_dict[1: 14]
f2 = data_dict[14: 29]
f3 = data_dict[29: 43]
f4 = data_dict[43: 57]
f5 = data_dict[57: 71]
f6 = data_dict[72: 83]

for f in range(1, 7):

    # Fold 1
    # if f == 1:
    #     dice_val_best = 0.0
    #     train_files = f1 + f2 + f3 + f4
    #     val_files = f5
    #     # test = f6
    #
    # # Fold 2
    # elif f == 2:
    #     dice_val_best = 0.0
    #     train_files = f1 + f2 + f3 + f6
    #     val_files = f4
    #     # test = f5
    #
    # # Fold 3
    # elif f == 3:
    #     dice_val_best = 0.0
    #     train_files = f1 + f2 + f5 + f6
    #     val_files = f3
    #     # test = f4
    #
    # # Fold 4
    # elif f == 4:
    #     dice_val_best = 0.0
    #     train_files = f1 + f4 + f5 + f6
    #     val_files = f2
    #     # test = f3
    #
    # # Fold 5
    # elif f == 5:
    #     dice_val_best = 0.0
    #     train_files = f3 + f4 + f5 + f6
    #     val_files = f1
    #     # test = f2
    #
    # # Fold 6
    # else:
    #     dice_val_best = 0.0
    #     train_files = f2 + f3 + f4 + f5
    #     val_files = f6
    #     # test = f1
    # train_ds = CacheDataset(
    #     data=train_files,
    #     transform=train_transforms,
    #     cache_num=24,
    #     cache_rate=1.0,
    #     num_workers=8,
    # )
    # train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=1, shuffle=True)
    # # DatasetSummary(train_ds, image_key='image_ct', label_key='image_sg').calculate_statistics()
    # val_ds = CacheDataset(
    #     data=val_files, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=4
    # )
    # val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)
    # # as explained in the "Setup transforms" section above, we want cached training images to not have metadata,
    # # and validations to have metadata the EnsureTyped transforms allow us to make this distinction on the other hand,
    # # set_track_meta is a global API; doing so here makes sure subsequent transforms (i.e., random transforms for
    # # training) will be carried out as Tensors, not MetaTensors
    # set_track_meta(True)
    #
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(
        spatial_dims=3,
        in_channels=6,
        out_channels=5,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2
    ).to(device)

    # weight = torch.load("model_log/model_swinvit.pt")
    # model.load_from(weights=weight)
    # print("Using pretrained self-supervied Swin UNETR backbone weights !")

    torch.backends.cudnn.benchmark = True
    loss_function = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler()

    # def validation(epoch_iterator_val):
    #     model.eval()
    #     with torch.no_grad():
    #         for step, batch in enumerate(epoch_iterator_val):
    #             val_inputs, val_labels = (batch["image_petct"].cuda(), batch["image_sg"].cuda())
    #             with torch.cuda.amp.autocast():
    #                 val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)
    #             val_labels_list = decollate_batch(val_labels)
    #             val_labels_convert = [
    #                 post_label(val_label_tensor) for val_label_tensor in val_labels_list
    #             ]
    #             val_outputs_list = decollate_batch(val_outputs)
    #             val_output_convert = [
    #                 post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
    #             ]
    #             dice_metric(y_pred=val_output_convert, y=val_labels_convert)
    #             epoch_iterator_val.set_description(
    #                 "Validate (%d / %d Steps)" % (global_step, 10.0)
    #             )
    #         mean_dice_val = dice_metric.aggregate().item()
    #         dice_metric.reset()
    #     return mean_dice_val


    # def train(global_step, train_loader, dice_val_best, global_step_best):
    #     model.train()
    #     epoch_loss = 0
    #     step = 0
    #     epoch_iterator = tqdm(
    #         train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    #     )
    #     for step, batch in enumerate(epoch_iterator):
    #         step += 1
    #         x, y = (batch["image_petct"].cuda(), batch["image_sg"].cuda())
    #         with torch.cuda.amp.autocast():
    #             logit_map = model(x)
    #             loss = loss_function(logit_map, y)
    #         scaler.scale(loss).backward()
    #         epoch_loss += loss.item()
    #         scaler.unscale_(optimizer)
    #         scaler.step(optimizer)
    #         scaler.update()
    #         optimizer.zero_grad()
    #         epoch_iterator.set_description(
    #             "Training (%d / %d Steps) (loss=%2.5f)"
    #             % (global_step, max_iterations, loss)
    #         )
    #         if (
    #             global_step % eval_num == 0 and global_step != 0
    #         ) or global_step == max_iterations:
    #             epoch_iterator_val = tqdm(
    #                 val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
    #             )
    #             dice_val = validation(epoch_iterator_val)
    #             epoch_loss /= step
    #             epoch_loss_values.append(epoch_loss)
    #             metric_values.append(dice_val)
    #             if dice_val > dice_val_best:
    #                 dice_val_best = dice_val
    #                 global_step_best = global_step
    #                 torch.save(
    #                     model.state_dict(), os.path.join(root_dir, f"best_metric_subregions_unet_model_f{f}.pth")
    #                 )
    #                 print(
    #                     "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
    #                         dice_val_best, dice_val
    #                     )
    #                 )
    #             else:
    #                 print(
    #                     "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
    #                         dice_val_best, dice_val
    #                     )
    #                 )
    #         global_step += 1
    #     return global_step, dice_val_best, global_step_best


    max_iterations = 45000
    eval_num = 500
    post_label = AsDiscrete(to_onehot=5)
    post_pred = AsDiscrete(argmax=True, to_onehot=5)
    post_pred_arg = AsDiscrete(argmax=True)

    dice_metric = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=False)
    global_step = 0
    dice_val_best = 0.0
    global_step_best = 0
    epoch_loss_values = []
    metric_values = []
    # while global_step < max_iterations:
    #     global_step, dice_val_best, global_step_best = train(
    #         global_step, train_loader, dice_val_best, global_step_best
    #     )
    model.load_state_dict(torch.load(os.path.join(root_dir, f"best_metric_subregions_unet_model_f{f}.pth")))

    print(
        f"train completed, best_metric: {dice_val_best:.4f} "
        f"at iteration: {global_step_best}"
    )


# test part
l_one = []
l_two = []
l_three = []
l_four = []


def time_test(epoch_iterator_test, model):
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_test):
            test_inputs, test_labels = (batch["image_petct"].cuda(), batch["image_sg"].cuda())
            with torch.cuda.amp.autocast():
                test_outputs = sliding_window_inference(test_inputs, (96, 96, 96), 4, model)
            test_labels_list = decollate_batch(test_labels)
            test_labels_convert = [
                post_label(test_label_tensor) for test_label_tensor in test_labels_list
            ]
            test_outputs_list = decollate_batch(test_outputs)
            test_output_convert = [
                post_pred_arg(test_pred_tensor) for test_pred_tensor in test_outputs_list
            ]
            # fillholes = [FillHoles(applied_labels=[1, 2, 3, 4], connectivity=3)(i) for i in test_output_convert]
            # largest = [KeepLargestConnectedComponent(applied_labels=[1, 2, 3, 4], is_onehot=True, connectivity=3)(i) for
            #            i in fillholes]
            # test_output_convert = [
            #     post_pred_arg(test_pred_tensor) for test_pred_tensor in largest
            # ]
            img = nib.Nifti1Image(batch['image_ct'][0, 0, :, :, :].cpu().detach().numpy(), affine=np.eye(4))
            seg = nib.Nifti1Image(batch['image_sg'][0, 0, :, :, :].cpu().detach().numpy(), affine=np.eye(4))
            pred = nib.Nifti1Image(test_output_convert[0].squeeze(axis=0).cpu().detach().numpy(), affine=np.eye(4))

            nib.save(img, f'unet_pred_subregions/ct_{step}_{f}.nii.gz')
            nib.save(seg, f'unet_pred_subregions/sg_{step}_{f}.nii.gz')
            nib.save(pred, f'unet_pred_subregions/pred_{step}_{f}.nii.gz')
        #     dice_metric(y_pred=test_output_convert, y=test_labels_convert)
        #     epoch_iterator_test.set_description(
        #         "Test (%d / %d Steps)" % (global_step, 10.0)
        #     )
        #     l_one.append(dice_metric.aggregate()[0].item())
        #     l_two.append(dice_metric.aggregate()[1].item())
        #     l_three.append(dice_metric.aggregate()[2].item())
        #     l_four.append(dice_metric.aggregate()[3].item())
        #
        # # mean_dice_test = dice_metric.aggregate().item()
        # back_dice = dice_metric.aggregate()[0].item()
        # first_dice = dice_metric.aggregate()[1].item()
        # second_dice = dice_metric.aggregate()[2].item()
        # th_dice = dice_metric.aggregate()[3].item()
        # # four_dice = dice_metric.aggregate()[4].item()
        #
        # print(back_dice, first_dice, second_dice, th_dice)
        # # mean_dice_test = (back_dice + first_dice + second_dice + th_dice) / 4
        # dice_metric.reset()


    # print('Aval', sum(back_dice_l)/len(back_dice_l))
    # print('Doyom', sum(first_dice_l)/len(first_dice_l))
    # print('Seyom', sum(second_dice_l)/len(second_dice_l))
    # print('Charom', sum(th_dice_l)/len(th_dice_l))

    # print('Mean Dice test', mean_dice_test)
    # return mean_dice_test


for f in range(1, 7):
    print('Fold:', f)
    slice_map = {
        "1/image.nii.gz": 170,
    }
    case_num = 4
    model.load_state_dict(torch.load(os.path.join(root_dir, f"best_metric_subregions_unet_model_f{f}.pth")))

    # Fold 1
    if f == 1:
        test_dicts = f6

    # Fold 2
    elif f == 2:
        test_dicts = f5

    # Fold 3
    elif f == 3:
        test_dicts = f4

    # Fold 4
    elif f == 4:
        test_dicts = f3

    # Fold 5
    elif f == 5:
        test_dicts = f2

    # Fold 6
    else:
        test_dicts = f1

    test_ds = CacheDataset(
        data=test_dicts, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=4
    )
    test_loader = ThreadDataLoader(test_ds, num_workers=0, batch_size=1)
    epoch_iterator_test = tqdm(
        test_loader, desc="Test (X / X Steps) (dice=X.X)", dynamic_ncols=True
    )
    time_test(epoch_iterator_test, model)
print(l_one, sum(l_one)/len(l_one))
print(l_two, sum(l_two)/len(l_two))
print(l_three, sum(l_three)/len(l_three))
print(l_four, sum(l_four)/len(l_four))