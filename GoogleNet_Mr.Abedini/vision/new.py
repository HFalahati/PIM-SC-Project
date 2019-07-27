import torchvision.models as models
import torch
googlenet = models.googlenet(pretrained=True)

from torchvision import transforms
transform = transforms.Compose([            #[1]
 transforms.Resize(256),                    #[2]
 transforms.CenterCrop(224),                #[3]
 transforms.ToTensor(),                     #[4]
 transforms.Normalize(                      #[5]
 mean=[0.485, 0.456, 0.406],                #[6]
 std=[0.229, 0.224, 0.225]                  #[7]
 )])

from PIL import Image
img = Image.open("test.jpg")

img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)

googlenet.eval()

out = googlenet(batch_t)
# print(out.shape)


with open('imageNetLabel.txt') as f:
    classes = [line.strip() for line in f.readlines()]
# _, index = torch.max(out, 1)


_, indices = torch.sort(out, descending=True)

percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

[print(classes[idx][9:], percentage[idx].item()) for idx in indices[0][:5]]

# print(classes[index[0]], percentage[index[0]].item())


img.close()