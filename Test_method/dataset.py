import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import open3d as o3d
import cv2 
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
import re
from fuzzywuzzy import fuzz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetaFood3DRGBDepthDataset(Dataset):

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        excel_file: str = None,
        image_size: Tuple[int, int] = (224, 224),
        max_points: int = 1024,
        augment: bool = True,
        use_depth: bool = True, 
        min_match_score: int = 80
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.max_points = max_points
        self.augment = augment
        self.use_depth = use_depth  # Store depth usage flag
        self.min_match_score = min_match_score
        self.split_dir = self.root_dir / split
        self.image_dir = self.split_dir / 'Blender_render_images'
        self.pointcloud_dir = self.split_dir / 'Point_cloud'

        # Load nutrition data
        self.nutrition_data = self._load_nutrition_data(
            excel_file if excel_file is not None
            else self.root_dir / '_MetaFood3D_new_complete_dataset_nutrition_v2 (1).xlsx')
        
        self.samples = self._build_sample_list()
        logger.info(f"Loaded {len(self.samples)} valid samples from {split} split")
        
        if self.use_depth:
            paired_count = sum(1 for s in self.samples if s.get('depth_file') is not None)
            logger.info(f"RGB-Depth paired: {paired_count}/{len(self.samples)} ({100*paired_count/len(self.samples):.1f}%)")

        self.image_transform = self._get_image_transforms()
        self.depth_transform = self._get_depth_transforms() 
        self.nutrition_stats = self._compute_nutrition_stats()

    def _load_nutrition_data(self, excel_file: Path) -> pd.DataFrame:
        df = pd.read_excel(excel_file)
        df['Food_Type_Clean'] = df['Food_Type'].str.lower().str.replace('_', ' ').str.strip()
        logger.info(f"Loaded nutrition data with {len(df)} entries")
        return df

    def _normalize_name(self, name: str) -> str:
        name = re.sub(r'_+', ' ', name.lower()) 
        name = re.sub(r'[^\w\s]', ' ', name)    
        name = ' '.join(name.split())           
        return name
    
    def _row_to_dict(self, row) -> Dict:
        return {
            'Object_name': row['Object_name'],
            'Food_Type': row['Food_Type'],
            'FNDDS Food Code': row['FNDDS Food Code'],
            'Weight (g)': float(row['Weight (g)']),
            'Energy (Kcal)': float(row['Energy (Kcal)']),
            'Protein (g)': float(row['Protein (g)']),
            'Fat (g)': float(row['Fat (g)']),
            'Carbs (g)': float(row['Carbs (g)']),
            'Volume': float(row['Volume'])
        }
    
    def _find_best_nutrition_match(self, food_instance_name: str) -> Optional[Dict]:
        normalized_name = self._normalize_name(food_instance_name)
        best_match, best_score = None, 0
        
        for _, row in self.nutrition_data.iterrows():
            if row['Food_Type'].lower() == food_instance_name.lower():
                return self._row_to_dict(row)
            
            clean_food_type = self._normalize_name(row['Food_Type'])
            score = fuzz.ratio(normalized_name, clean_food_type)
            if score > best_score and score >= self.min_match_score:
                best_score = score
                best_match = row
        
        if best_match is not None:
            logger.debug(f"Matched '{food_instance_name}' with '{best_match['Food_Type']}' (score: {best_score})")
            return self._row_to_dict(best_match)
        
        return None

    def _find_rgb_depth_pairs(self, original_dir: Path, depth_dir: Path) -> List[Tuple[Path, Optional[Path]]]:
        pairs = []
        rgb_files = list(original_dir.glob('*.png'))
        if not rgb_files:
            return pairs
        
        depth_mapping = {}
        if self.use_depth and depth_dir.exists():
            depth_files = list(depth_dir.glob('*.png'))
            for depth_file in depth_files:
                depth_stem = depth_file.stem 
                if '_depth0001' in depth_stem:
                    rgb_base = depth_stem.replace('_depth0001', '')
                    depth_mapping[rgb_base] = depth_file
                else:
                    depth_mapping[depth_stem] = depth_file
        
        for rgb_file in rgb_files:
            rgb_stem = rgb_file.stem  # Remove .png extension
            depth_path = None
            if self.use_depth:
                depth_path = depth_mapping.get(rgb_stem)
                if depth_path is None:
                    depth_key = f"{rgb_stem}_depth0001"
                    depth_path = depth_mapping.get(depth_key)
            
            pairs.append((rgb_file, depth_path))
            
            if depth_path is not None:
                logger.debug(f"Paired: {rgb_file.name} <-> {depth_path.name}")
            elif self.use_depth:
                logger.debug(f"No depth match for: {rgb_file.name}")
        
        return pairs

    def _build_sample_list(self) -> List[Dict]:
        samples = []
        
        for food_category in self.image_dir.iterdir():
            if not food_category.is_dir():
                continue
            food_category_name = food_category.name
            
            for food_instance in food_category.iterdir():
                if not food_instance.is_dir():
                    continue
                food_instance_name = food_instance.name
                pc_path = self.pointcloud_dir / food_category_name / food_instance_name
                if not pc_path.exists():
                    continue
                ply_files = list(pc_path.glob('*.ply'))
                if not ply_files:
                    continue
                nutrition_info = self._find_best_nutrition_match(food_instance_name)
                if nutrition_info is None:
                    continue
                food_data_dir = food_instance / food_instance_name
                if not food_data_dir.exists():
                    continue
                original_dir = food_data_dir / 'Original'
                if not original_dir.exists():
                    continue
                depth_dir = food_data_dir / 'Depth' 
                rgb_depth_pairs = self._find_rgb_depth_pairs(original_dir, depth_dir)
                if not rgb_depth_pairs:
                    continue
                
                for rgb_file, depth_file in rgb_depth_pairs:
                    samples.append({
                        'food_category': food_category_name,
                        'food_instance': food_instance_name,
                        'rgb_file': rgb_file,           
                        'depth_file': depth_file,     
                        'pointcloud_file': ply_files[0],
                        'nutrition': nutrition_info,
                        'calories': nutrition_info['Energy (Kcal)'],
                        'weight': nutrition_info['Weight (g)'],
                        'volume': nutrition_info['Volume']
                    })
        
        return samples

    def _get_image_transforms(self):
        """RGB image transforms"""
        if self.augment and self.split == 'train':
            return transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _get_depth_transforms(self):
        """Depth image transforms (no normalization, just resize and convert to tensor)"""
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ])

    def _compute_nutrition_stats(self) -> Dict:
        calories = [s['calories'] for s in self.samples]
        weights = [s['weight'] for s in self.samples]
        volumes = [s['volume'] for s in self.samples]
        proteins = [s['nutrition']['Protein (g)'] for s in self.samples]
        fats = [s['nutrition']['Fat (g)'] for s in self.samples]
        carbs = [s['nutrition']['Carbs (g)'] for s in self.samples]
        
        return {
            'calories': {'mean': np.mean(calories), 'std': np.std(calories)},
            'weight': {'mean': np.mean(weights), 'std': np.std(weights)},
            'volume': {'mean': np.mean(volumes), 'std': np.std(volumes)},
            'protein': {'mean': np.mean(proteins), 'std': np.std(proteins)},
            'fat': {'mean': np.mean(fats), 'std': np.std(fats)},
            'carbs': {'mean': np.mean(carbs), 'std': np.std(carbs)}
        }

    def _load_pointcloud(self, pc_path: Path) -> np.ndarray:
        pcd = o3d.io.read_point_cloud(str(pc_path))
        points = np.asarray(pcd.points)
        if len(points) == 0:
            logger.warning(f"No point cloud: {pc_path}")
            return np.zeros((self.max_points, 3))
        centroid = np.mean(points, axis=0)
        points = points - centroid
        distances = np.linalg.norm(points, axis=1)
        max_distance = np.max(distances)
        if max_distance > 0:
            points = points / max_distance
        num_points = len(points)
        if num_points >= self.max_points:
            indices = np.random.choice(num_points, self.max_points, replace=False)
            return points[indices]
        else:
            pad_size = self.max_points - num_points
            pad_indices = np.random.choice(num_points, pad_size, replace=True)
            return np.concatenate([points, points[pad_indices]], axis=0)


    def _load_rgb_image(self, rgb_file: Path) -> Image.Image:
        return Image.open(rgb_file).convert('RGB')


    def _load_depth_image(self, depth_file: Optional[Path]) -> torch.Tensor:
        if depth_file is None or not depth_file.exists():
            return torch.zeros(1, *self.image_size)
        

        depth_img = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
        
        if depth_img is None:
            logger.warning(f"Could not load depth image: {depth_file}")
            return torch.zeros(1, *self.image_size)
        
        if len(depth_img.shape) == 3:
            depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)
        
        depth_pil = Image.fromarray(depth_img)
        depth_tensor = self.depth_transform(depth_pil)
        if depth_tensor.max() > 1.0:
            if depth_tensor.max() <= 255:
                depth_tensor = depth_tensor / 255.0  # 8-bit
            else:
                depth_tensor = depth_tensor / 65535.0  # 16-bit
        
        return depth_tensor
            


    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        rgb_image = self._load_rgb_image(sample['rgb_file'])
        depth_image = self._load_depth_image(sample['depth_file'])
        pointcloud = self._load_pointcloud(sample['pointcloud_file'])
        
        return {
            'image': self.image_transform(rgb_image),                  
            'depth': depth_image,                                   
            'pointcloud': torch.from_numpy(pointcloud).float(),       
            'calories': torch.tensor(sample['calories'], dtype=torch.float32),
            'weight': torch.tensor(sample['weight'], dtype=torch.float32),
            'volume': torch.tensor(sample['volume'], dtype=torch.float32),
            'protein': torch.tensor(sample['nutrition']['Protein (g)'], dtype=torch.float32),
            'fat': torch.tensor(sample['nutrition']['Fat (g)'], dtype=torch.float32),
            'carbs': torch.tensor(sample['nutrition']['Carbs (g)'], dtype=torch.float32),
            'food_category': sample['food_category'],
            'food_instance': sample['food_instance'],
            'rgb_filename': sample['rgb_file'].name,
            'depth_filename': sample['depth_file'].name if sample['depth_file'] else 'None'
        }


class MetaFood3DRGBDepthDataModule:
    
    def __init__(
        self,
        root_dir: str,
        excel_file: str = None,
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: Tuple[int, int] = (224, 224),
        max_points: int = 1024,
        use_depth: bool = True, 
        min_match_score: int = 80
    ):
        self.root_dir = root_dir
        self.excel_file = excel_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.max_points = max_points
        self.use_depth = use_depth

        self.train_dataset = MetaFood3DRGBDepthDataset(
            root_dir=root_dir,
            split='train',
            excel_file=excel_file,
            image_size=image_size,
            max_points=max_points,
            augment=True,
            use_depth=use_depth,
            min_match_score=min_match_score
        )
        
        self.nutrition_stats = self.train_dataset.nutrition_stats
        logger.info(f"Nutrition stats from train data: {self.nutrition_stats}")
        
        self.test_dataset = MetaFood3DRGBDepthDataset(
            root_dir=root_dir,
            split='test',
            excel_file=excel_file,
            image_size=image_size,
            max_points=max_points,
            augment=False,
            use_depth=use_depth,
            min_match_score=min_match_score
        )
        self.test_dataset.nutrition_stats = self.nutrition_stats

    def get_train_loader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def get_test_loader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )

    def get_nutrition_stats(self) -> Dict:
        return self.nutrition_stats

    def verify_rgb_depth_pairing(self, num_samples: int = 5):
        logger.info(f" Verifying RGB-Depth pairing (showing {num_samples} samples):")
        
        for i in range(min(num_samples, len(self.train_dataset))):
            sample = self.train_dataset[i]
            logger.info(f"  Sample {i+1}:")
            logger.info(f"    RGB: {sample['rgb_filename']}")
            logger.info(f"    Depth: {sample['depth_filename']}")
            logger.info(f"    Category: {sample['food_category']}")
            logger.info(f"    Instance: {sample['food_instance']}")
            logger.info(f"    Calories: {sample['calories'].item():.2f}")
            logger.info("")



def create_rgbd_4channel_tensor(rgb_tensor: torch.Tensor, depth_tensor: torch.Tensor) -> torch.Tensor:
    return torch.cat([rgb_tensor, depth_tensor], dim=1)


if __name__ == "__main__":
    ROOT_DIR = r"E:\MetaFood3D_Calo"
    EXCEL_FILE = r"E:\MetaFood3D_Calo\_MetaFood3D_new_complete_dataset_nutrition_v2 (1).xlsx"
    print(" Testing RGB + Depth Dataset...")
    data_module = MetaFood3DRGBDepthDataModule(
        ROOT_DIR, 
        EXCEL_FILE, 
        batch_size=4,
        num_workers=0, 
        use_depth=True
    )
    print(f" Train samples: {len(data_module.train_dataset)}")
    print(f" Test samples: {len(data_module.test_dataset)}")
    print(" Nutrition stats:", data_module.get_nutrition_stats())
    data_module.verify_rgb_depth_pairing(num_samples=3)
    train_loader = data_module.get_train_loader()
    batch = next(iter(train_loader))
    
    print("\n Sample batch shapes:")
    print(f"  RGB: {batch['image'].shape}")    
    print(f"  Depth: {batch['depth'].shape}")      
    print(f"  Point Cloud: {batch['pointcloud'].shape}")  
    print(f"  Calories: {batch['calories'].shape}")
    
    rgbd_4channel = create_rgbd_4channel_tensor(batch['image'], batch['depth'])
    print(f"  RGBD (4ch): {rgbd_4channel.shape}")  #  [B, 4, H, W]
    print("\n RGB + Depth dataset loading successful")
    print(" Ready for 'Depth as 4th Channel' approach!")
