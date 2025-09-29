import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import open3d as o3d
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
import re
from fuzzywuzzy import fuzz
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetaFood3DDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        excel_file: str = None,
        image_size: Tuple[int, int] = (224, 224),
        max_points: int = 1024,
        augment: bool = True,
        min_match_score: int = 80
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.max_points = max_points
        self.augment = augment
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

        self.image_transform = self._get_image_transforms()
        self.nutrition_stats = self._compute_nutrition_stats()

    def _load_nutrition_data(self, excel_file: Path) -> pd.DataFrame:
        try:
            df = pd.read_excel(excel_file)
            df['Food_Type_Clean'] = df['Food_Type'].str.lower().str.replace('_', ' ').str.strip()
            logger.info(f"Loaded nutrition data with {len(df)} entries")
            return df
        except Exception as e:
            logger.error(f"Error loading nutrition data: {e}")
            raise

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
                #print('yes')
                return self._row_to_dict(row)
            clean_food_type = self._normalize_name(row['Food_Type'])
            score = fuzz.ratio(normalized_name, clean_food_type)
            if score > best_score and score >= self.min_match_score:
                best_score = score
                best_match = row
        if best_match is not None:
            logger.info(f"Matched '{food_instance_name}' with '{best_match['Food_Type']}' (score: {best_score})")
            return self._row_to_dict(best_match)
        ## new_adammame_4

 

    def _build_sample_list(self) -> List[Dict]:
        samples = []
        for food_category in self.image_dir.iterdir():
            if not food_category.is_dir():continue
            food_category_name = food_category.name
            for food_instance in food_category.iterdir():
                if not food_instance.is_dir():continue
                food_instance_name = food_instance.name
                pc_path = self.pointcloud_dir / food_category_name / food_instance_name
                if not pc_path.exists():continue
                ply_files = list(pc_path.glob('*.ply'))
                if not ply_files:continue
                nutrition_info = self._find_best_nutrition_match(food_instance_name)
                if nutrition_info is None:continue
                food_data_dir = food_instance / food_instance_name
                if not food_data_dir.exists():continue
                original_dir = food_data_dir / 'Original'
                if not original_dir.exists():continue
                image_files = list(original_dir.glob('*.png'))
                if not image_files:continue
                samples.append({
                    'food_category': food_category_name,
                    'food_instance': food_instance_name,
                    'image_files': image_files,
                    'pointcloud_file': ply_files[0],
                    'nutrition': nutrition_info,
                    'calories': nutrition_info['Energy (Kcal)'],
                    'weight': nutrition_info['Weight (g)'],
                    'volume': nutrition_info['Volume']
                })
        return samples

    def _get_image_transforms(self):
        if self.augment and self.split == 'train':
            return transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])#giá trị trung bình và độ lệch chuẩn của dataset ImageNet
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
        points = np.asarray(pcd.points)#[num_points,3]
        if len(points) == 0:
            logger.warning(f"No point cloud: {pc_path}")
            return np.zeros((self.max_points, 3))
        centroid = np.mean(points, axis=0)
        points = points - centroid#dịch chuyển về tọa độ 0 0 0
        distances = np.linalg.norm(points, axis=1)
        max_distance = np.max(distances)
        if max_distance > 0:
            points = points / max_distance #[-1,1]
        num_points = len(points)
        if num_points >= self.max_points:
            indices = np.random.choice(num_points, self.max_points, replace=False)
            return points[indices]#[self.max_points, 3]
        pad_size = self.max_points - num_points
        pad_indices = np.random.choice(num_points, pad_size, replace=True)
        return np.concatenate([points, points[pad_indices]], axis=0)# self.max_points,3
        

    def _load_random_image(self, image_files: List[Path]):
        image_path = np.random.choice(image_files)
        return Image.open(image_path).convert('RGB')
        


    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        image = self._load_random_image(sample['image_files'])
        return {
            'image': self.image_transform(image),
            'pointcloud': torch.from_numpy(self._load_pointcloud(sample['pointcloud_file'])),
            'calories': torch.tensor(sample['calories'], dtype=torch.float32),
            'weight': torch.tensor(sample['weight'], dtype=torch.float32),
            'volume': torch.tensor(sample['volume'], dtype=torch.float32),
            'protein': torch.tensor(sample['nutrition']['Protein (g)'], dtype=torch.float32),
            'fat': torch.tensor(sample['nutrition']['Fat (g)'], dtype=torch.float32),
            'carbs': torch.tensor(sample['nutrition']['Carbs (g)'], dtype=torch.float32),
            'food_category': sample['food_category'],
            'food_instance': sample['food_instance']
        }

class MetaFood3DDataModule:
    def __init__(
        self,
        root_dir: str,
        excel_file: str = None,
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: Tuple[int, int] = (224, 224),
        max_points: int = 1024,
        min_match_score: int = 80
    ):
        self.root_dir = root_dir
        self.excel_file = excel_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.max_points = max_points

        self.train_dataset = MetaFood3DDataset(
            root_dir=root_dir,
            split='train',
            excel_file=excel_file,
            image_size=image_size,
            max_points=max_points,
            augment=True,
            min_match_score=min_match_score
        )
        self.nutrition_stats = self.train_dataset.nutrition_stats
        logger.info(f"Nutrition stats train data: {self.nutrition_stats}")
        self.test_dataset = MetaFood3DDataset(
            root_dir=root_dir,
            split='test',
            excel_file=excel_file,
            image_size=image_size,
            max_points=max_points,
            augment=False,
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

if __name__ == "__main__":
    ROOT_DIR = r"E:\MetaFood3D_Calo"
    EXCEL_FILE = r"E:\MetaFood3D_Calo\_MetaFood3D_new_complete_dataset_nutrition_v2 (1).xlsx"
    data_module = MetaFood3DDataModule(ROOT_DIR, EXCEL_FILE)
    print(f"Train samples: {len(data_module.train_dataset)}")
    print(f"Test samples: {len(data_module.test_dataset)}")
    print("Nutrition stats:", data_module.get_nutrition_stats())
    print("\n successfully!")
