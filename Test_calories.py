# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# import matplotlib.pyplot as plt
# from models.rgb2point_model import FoodCalorieEstimationModel

# image_path = "E:\\WindowsApps\\Downloads\\garan.jpg"

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = FoodCalorieEstimationModel().to(device)
# checkpoint = torch.load("E:\WindowsApps\\Downloads\\best_calorie.pth", 
#                        map_location=device, weights_only=False)
# model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], 
#                         std=[0.229, 0.224, 0.225])
# ])

# image = Image.open(image_path).convert('RGB')
# input_tensor = transform(image).unsqueeze(0).to(device)

# with torch.no_grad():
#     output = model(input_tensor)
#     nutrition_stats = checkpoint['nutrition_stats']
#     pred_norm = output['calories'].item()
#     calories = pred_norm * nutrition_stats['calories']['std'] + nutrition_stats['calories']['mean']

# plt.figure(figsize=(10, 8))
# plt.imshow(image)
# plt.axis('off')
# plt.title(f' Calories dự đoán: {calories:.2f} kcal', 
#           fontsize=16, fontweight='bold', pad=50)
# plt.tight_layout()
# plt.show()

# print(f" Calo dự đoán: {calories:.2f} kcal")
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import torchvision.transforms as transforms
from models.rgb2point_model import FoodCalorieEstimationModel

yolo = YOLO(r"E:\WindowsApps\Downloads\best.pt")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cal_model = FoodCalorieEstimationModel().to(device)
ckpt = torch.load(r"E:\MetaFood3D_Calo\calorie_checkpoints\best_calorie.pth", map_location=device)
cal_model.load_state_dict(ckpt['model_state_dict']); cal_model.eval()
stats = ckpt['nutrition_stats']

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def denorm(x,key): return x*stats[key]['std']+stats[key]['mean']

img_path = r"E:\WindowsApps\Downloads\xucxichtrung.jpg"
res = yolo.predict(img_path, conf=0.3)[0]

if res.boxes:
    best = max([{"box":b.xyxy.cpu().numpy()[0],"cls":int(b.cls),"conf":float(b.conf)} for b in res.boxes], key=lambda x:x["conf"])
    names = yolo.model.names
    pil = Image.open(img_path).convert("RGB")
    x1,y1,x2,y2 = map(int,best["box"]); crop = pil.crop((x1,y1,x2,y2))
    inp = transform(crop).unsqueeze(0).to(device)
    with torch.no_grad():
        out = cal_model(inp)
        cal = denorm(out['calories'].item(),'calories')
        w = denorm(out['weight'].item(),'weight') if 'weight' in stats else None

    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()

    draw.rectangle([x1,y1,x2,y2], outline="red", width=3)
    text = f"{names[best['cls']]} {best['conf']:.2f}\n calo dự đoán {cal:.1f} kcal"
    if w: text += f" | weight {w:.1f} g"

    text_y = max(y1-60, 0)  
    draw.text((x1, text_y), text, fill="red", font=font)

    plt.figure(figsize=(10,8))
    plt.imshow(np.array(pil))  
    plt.axis("off"); plt.show()
else:
    print("Không phát hiện đối tượng.")
