import torch
import torch.nn as nn

from kornia.color.lab import rgb_to_lab, lab_to_rgb
# from kornia.color.lab import rgb_to_hsv, hsv_to_rgb

class ColorTransformer(object):
    def get_mean_std(self, img, lab=False, frame=False):
        img = rgb_to_lab(img) if lab else img
        num_img = len(img)
        if frame:
            img = torch.cat([img[:, :, :20].reshape(num_img, 3, -1), img[:, :, -70:].reshape(num_img, 3, -1),
                             img[:, :, :, :20].reshape(num_img, 3, -1), img[:, :, :, -20:].reshape(num_img, 3, -1)],
                            dim=2)
            means, stds = torch.mean(img, 2), torch.std(img, 2)
        else:
            means, stds = torch.mean(img, (2, 3)), torch.std(img, (2, 3))
        return means, stds

    def get_mean_std_dist(self, img, lab=False, frame=False):
        means, stds = self.get_mean_std(img, lab, frame)
        return torch.mean(means, 0), torch.std(means, 0), torch.mean(stds, 0), torch.std(stds, 0)

    def dist_resample(self, img, mean_mean, mean_std, std_mean, std_std):
        mean = torch.normal(mean_mean.expand(img.shape[0], 3), mean_std.expand(img.shape[0], 3))
        std = torch.normal(std_mean.expand(img.shape[0], 3), std_std.expand(img.shape[0], 3))
        return mean, std

    def normalize_target(self, img, src_mean, src_std, tgt_mean, tgt_std):
        num_img = len(img)
        img = (img - src_mean.reshape(num_img, 3, 1, 1)) / src_std.reshape(num_img, 3, 1, 1)
        img = img * tgt_std.reshape(num_img, 3, 1, 1) + tgt_mean.reshape(num_img, 3, 1, 1)
        return img

    def color_transfer_target(self, img, mean, std, lab=True, frame=False):
        img = rgb_to_lab(img) if lab else img
        src_mean, src_std = self.get_mean_std(img, frame=frame)
        img = self.normalize_target(img, src_mean, src_std, mean, std)
        return lab_to_rgb(img) if lab else img

    def color_transfer_resample(self, img, mean_mean=None, mean_std=None, std_mean=None, std_std=None, lab=True, frame=False):
        img = rgb_to_lab(img) if lab else img
        src_mean, src_std = self.get_mean_std(img, frame)
        if mean_mean is None:
            mean_mean, mean_std, std_mean, std_std = self.get_mean_std_dist(img)
        elif mean_std is None:
            mean_std = torch.tensor([0., 0., 0.]).cuda()
            std_std = torch.tensor([0., 0., 0.]).cuda()
        mean, std = self.dist_resample(img, mean_mean, mean_std, std_mean, std_std)
        img = self.normalize_target(img, src_mean, src_std, mean, std)
        return lab_to_rgb(img) if lab else img

    def color_transfer_shuffle(self, img, lab=True, frame=False):
        img = rgb_to_lab(img) if lab else img
        src_mean, src_std = self.get_mean_std(img, frame)
        rand_indices = torch.randperm(len(src_mean))
        mean = src_mean[rand_indices]
        std = src_std[rand_indices]
        img = self.normalize_target(img, src_mean, src_std, mean, std)
        return lab_to_rgb(img) if lab else img
