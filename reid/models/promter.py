import torch
import torch.nn as nn


class KernelLearning(nn.Module):
    def __init__(self,n_kernel,groups=3, G_lr=2e-4, G_B1=0.0, G_B2=0.999, adam_eps=1e-8, model='shufflenet_v2'):
        super(KernelLearning, self).__init__()
        
        # model='mobilenet_v3'

        if model=='shufflenet_v2':
            shufflenet = torch.hub.load('pytorch/vision:v0.10.0', 'shufflenet_v2_x0_5', pretrained=True)
            self.backbone = nn.Sequential(
                shufflenet.conv1, shufflenet.maxpool,
                shufflenet.stage2, shufflenet.stage3, shufflenet.stage4, shufflenet.conv5
            )
            n_dims = shufflenet.fc.in_features
            # print(shufflenet)
        else:
            import torchvision
            mobilenet_v3=torchvision.models.mobilenetv3.mobilenet_v3_small(pretrained=True)
            self.backbone=mobilenet_v3.features
            n_dims = mobilenet_v3.classifier[0].in_features
        # exit(0)
        self.gap = nn.AdaptiveMaxPool2d(1)

        
        if 3==groups:
            out_dim=n_kernel*(3*3*3+3)
        else:
            out_dim=n_kernel*(3*3*3*3+3)
        self.kernel_predicter = nn.Sequential(
            nn.Linear(n_dims, n_dims // 2),
            nn.ReLU(),
            nn.Linear(n_dims // 2, out_dim)
        )
       
        self.init_weights()
        self.lr = G_lr
        self.B1 = G_B1
        self.B2 = G_B2
        self.adam_eps = adam_eps
        self.optim = torch.optim.Adam(params=self.parameters(), lr=self.lr,
                                      betas=(self.B1, self.B2), weight_decay=0,
                                      eps=self.adam_eps)

    def forward(self, x, return_feature=False):
        x = self.backbone(x)
        x = self.gap(x)
        x = x.view(x.shape[0], -1)
        if return_feature:
            return x
        else:
            # return self.mean_gamma_layers(x), self.mean_bias_layers(x), self.std_gamma_layers(x), self.std_bias_layers(x)
            return self.kernel_predicter(x)

    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d)
                or isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print('Param count for Prompter''s initialized parameters: %d' % self.param_count)

