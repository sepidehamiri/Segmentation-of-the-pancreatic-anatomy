import os
import shutil
import tempfile

import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd, SaveImaged,
    RandRotate90d, AddChanneld,
    EnsureTyped, ConcatItemsd,
    SplitChannel, Activationsd,
    Invertd, AsDiscreted,
    KeepLargestConnectedComponent, FillHoles
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR

from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    set_track_meta, NiftiSaver
)

import nibabel as nib
import numpy as np
import torch

print_config()

directory = "model_log/swin_unet"
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

pre_transforms = Compose(
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
        # Orientationd(keys=keys, axcodes="LAS"),
        # ConcatItemsd(keys=["image_pt", "image_ct"], name="image_petct", dim=0),

        # EnsureTyped(keys=["image_petct", "image_sg"], device=device),
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

    model = SwinUNETR(
        img_size=(64, 64, 64),
        in_channels=6,
        out_channels=5,
        feature_size=48,
        use_checkpoint=True,
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
    #                 val_outputs = sliding_window_inference(val_inputs, (64, 64, 64), 4, model)
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
    #
    #
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
    #                     model.state_dict(), os.path.join(root_dir, f"best_metric_model_swin_f{f}.pth")
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
    eval_num = 400
    post_label = AsDiscrete(to_onehot=5)
    post_pred = AsDiscrete(argmax=True, to_onehot=5)
    # post_transforms = Compose([
    #     AsDiscrete(argmax=True, to_onehot=5),
    #     SaveImage(output_dir="./out", output_postfix="seg", resample=False),
    # ])
    # post_pred = Compose([
    #     Activationsd(keys="pred", softmax=True),
    #     Invertd(
    #         keys="pred",  # invert the `pred` data field, also support multiple fields
    #         transform=pre_transforms,
    #         orig_keys="petct",  # get the previously applied pre_transforms information on the `img` data field,
    #                           # then invert `pred` based on this information. we can use same info
    #                           # for multiple fields, also support different orig_keys for different fields
    #         nearest_interp=False,  # don't change the interpolation mode to "nearest" when inverting transforms
    #                                # to ensure a smooth output, then execute `AsDiscreted` transform
    #         to_tensor=True,  # convert to PyTorch Tensor after inverting
    #     ),
    #     AsDiscreted(keys="pred", threshold=0.5),
    #     SaveImaged(keys="pred", output_dir="./out", output_postfix="seg", resample=False),
    # ])
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
    model.load_state_dict(torch.load(os.path.join(root_dir, f"best_metric_model_swin_f{f}.pth")))

    print(
        f"train completed, best_metric: {dice_val_best:.4f} "
        f"at iteration: {global_step_best}"
    )


slice_map = {
    "img0035.nii.gz": 170,
    "img0036.nii.gz": 230,
    "img0037.nii.gz": 204,
    "img0038.nii.gz": 204,
    "img0039.nii.gz": 204,
    "img0040.nii.gz": 180,
}


# test part
def time_test(epoch_iterator_test, model, f):
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_test):
            test_inputs, test_labels = (batch["image_petct"].cuda(), batch["image_sg"].cuda())
            with torch.cuda.amp.autocast():
                test_outputs = sliding_window_inference(test_inputs, (64, 64, 64), 4, model)
            test_labels_list = decollate_batch(test_labels)
            test_labels_convert = [
                post_label(test_label_tensor) for test_label_tensor in test_labels_list
            ]
            test_outputs_list = decollate_batch(test_outputs)
            test_output_convert = [
                post_pred(test_pred_tensor) for test_pred_tensor in test_outputs_list
            ]
            fillholes = [FillHoles(applied_labels=[1, 2, 3, 4], connectivity=3)(i) for i in test_output_convert]
            largest = [KeepLargestConnectedComponent(applied_labels=[1, 2, 3, 4], is_onehot=True, connectivity=3)(i) for i in fillholes]
            img = nib.Nifti1Image(batch['image_ct'][0, 0, :, :, :].cpu().detach().numpy(), affine=np.eye(4))
            seg = nib.Nifti1Image(batch['image_sg'][0, 0, :, :, :].cpu().detach().numpy(), affine=np.eye(4))
            pred = nib.Nifti1Image(largest[0].squeeze(axis=0).cpu().detach().numpy(), affine=np.eye(4))

            nib.save(img, f'ct_{step}_{f}.nii.gz')
            nib.save(seg, f'sg_{step}_{f}.nii.gz')
            nib.save(pred, f'pred_{step}_{f}.nii.gz')

            # plt.figure("check", (18, 6))
            # plt.subplot(1, 3, 1)
            # plt.title(f"image")
            # plt.imshow(batch["image_ct"][0, 0, :, :, 90], cmap="gray")
            # plt.subplot(1, 3, 2)
            # plt.title(f"label")
            # plt.imshow(batch["image_sg"][0, 0, :, :, 90].cpu())
            # plt.subplot(1, 3, 3)
            # plt.title(f"output")
            # plt.imshow(torch.argmax(
            #     test_output_convert[0], dim=0).detach().cpu()[:, :, 90])
            # plt.show()
            dice_metric(y_pred=test_output_convert, y=test_labels_convert)
            epoch_iterator_test.set_description(
                "Test (%d / %d Steps)" % (global_step, 10.0)
            )

        # mean_dice_test = dice_metric.aggregate().item()
        back_dice = dice_metric.aggregate()[0].item()
        first_dice = dice_metric.aggregate()[1].item()
        second_dice = dice_metric.aggregate()[2].item()
        th_dice = dice_metric.aggregate()[3].item()
        # four_dice = dice_metric.aggregate()[4].item()
        print(back_dice, first_dice, second_dice, th_dice)
        mean_dice_test = (back_dice + first_dice + second_dice + th_dice) / 5
        dice_metric.reset()
    print('Mean Dice test', mean_dice_test)
    return mean_dice_test


for f in range(1, 7):
    print('Fold:', f)
    slice_map = {
        "1/image.nii.gz": 170,
    }
    case_num = 4
    model.load_state_dict(torch.load(os.path.join(root_dir, f"best_metric_model_swin_f{f}.pth")))

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
    time_test(epoch_iterator_test, model, f)
