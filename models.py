import torchvision
import torch.nn as nn

class Video_Classification_Models(nn.Module):
    def __init__(self, model_name, num_classes):
        super(Video_Classification_Models, self).__init__()
        self.model_name = model_name
        #self.model = torchvision.models.video.r2plus1d_18(pretrained=True, progress=True)
        # self.model = torchvision.models.video.mc3_18(pretrained=True, progress=True)
        # self.model = torchvision.models.video.r3d_18(pretrained=True, progress=True)

        if self.model_name == "r3d":
            self.model = torchvision.models.video.r3d_18(weights=torchvision.models.video.R3D_18_Weights.KINETICS400_V1, progress=True)
            # print("Model (self.model_name):", self.model)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif self.model_name == "r2plus1d":
            self.model = torchvision.models.video.r2plus1d_18(weights=torchvision.models.video.R2Plus1D_18_Weights.KINETICS400_V1, progress=True)
            # print("Model (self.model_name):", self.model)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif self.model_name == "mc3":
            self.model = torchvision.models.video.mc3_18(weights=torchvision.models.video.MC3_18_Weights.KINETICS400_V1, progress=True)
            # print("Model (self.model_name):", self.model)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif self.model_name == "s3d":
            self.model = torchvision.models.video.s3d(weights=torchvision.models.video.S3D_Weights.KINETICS400_V1, progress=True)
            # print("Model (self.model_name):", self.model)
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Conv3d(1024, num_classes, kernel_size=1, stride=1, bias=True)
            )
        elif self.model_name == "mvit_v1":
            self.model = torchvision.models.video.mvit_v1_b(weights=torchvision.models.video.MViT_V1_B_Weights.KINETICS400_V1, progress=True)
            # print("Model (self.model_name):", self.model)
            self.model.head = nn.Sequential(
                nn.Dropout(p=self.model.head[0].p),
                nn.Linear(in_features=self.model.head[1].in_features, out_features=num_classes, bias=True)
            )
        elif self.model_name == "mvit_v2":
            self.model = torchvision.models.video.mvit_v2_s(weights=torchvision.models.video.MViT_V2_S_Weights.KINETICS400_V1, progress=True)
            # print("Model (self.model_name):", self.model)
            # print("self.model.head[0].p:", self.model.head[0].p)
            # print("self.model.head[1].in_features:", self.model.head[1].in_features)
            self.model.head = nn.Sequential(
                nn.Dropout(p=self.model.head[0].p),
                nn.Linear(in_features=self.model.head[1].in_features, out_features=num_classes, bias=True)
            )


    def forward(self, x):
        return self.model.forward(x)

"""
if __name__ == '__main__':

    # model = Video_Classification_Models(model_name="r3d", num_classes=3)
    # model = Video_Classification_Models(model_name="r2plus1d", num_classes=3)
    # model = Video_Classification_Models(model_name="mc3", num_classes=3)
    # model = Video_Classification_Models(model_name="s3d", num_classes=3)
    # model = Video_Classification_Models(model_name="mvit_v1", num_classes=3)
    model = Video_Classification_Models(model_name="mvit_v2", num_classes=3)

    print("New Model :", model)
"""