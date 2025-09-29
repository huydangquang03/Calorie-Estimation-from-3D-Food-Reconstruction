import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from models.rgb2point_model import RGB2PointModel

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image.unsqueeze(0) 
def normalize_point_cloud(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    max_dist = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / max_dist
    return pc

def visualize(image_tensor, pred_pc, save_path='test_visualization.png'):
    img = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)

    pred_pc_np = pred_pc.squeeze(0).cpu().numpy()  # [num_points, 3]
    #pred_pc_np = normalize_point_cloud(pred_pc_np)

    fig = plt.figure(figsize=(15, 5))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(img)
    ax1.set_title('Input Image')
    ax1.axis('off')

    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.scatter(pred_pc_np[:, 0], pred_pc_np[:, 1], pred_pc_np[:, 2], c=pred_pc_np[:, 2], cmap='viridis', s=5)
    ax2.set_title('Predicted Point Cloud')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.view_init(elev=20, azim=45) 
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.scatter(pred_pc_np[:, 0], pred_pc_np[:, 1], pred_pc_np[:, 2], c=pred_pc_np[:, 2], cmap='viridis', s=5)
    ax3.set_title('Predicted Point Cloud (Side View)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.view_init(elev=0, azim=90)  
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Visualization saved to {save_path}")

def main():
    image_path = r"E:\WindowsApps\Downloads\hotbot.jpg"
    checkpoint_path = r"E:\MetaFood3D_Calo\checkpoints\best.pth" 
    num_points = 1024  
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  
    output_vis = r"E:\MetaFood3D_Calo\point_cloud.png"  
    # Load model
    model = RGB2PointModel(num_points=num_points)
    model.to(device)
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    image = load_image(image_path).to(device)
    # Generate point cloud
    with torch.no_grad():
        pred_pointcloud = model(image)  
    visualize(image, pred_pointcloud, save_path=output_vis)
    
if __name__ == "__main__":
    main()