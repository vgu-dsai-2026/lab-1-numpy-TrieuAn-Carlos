from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from lab_utils.visualization import plot_feature_vector, show_image_gallery
LABELS = ('cat', 'dog')
LABEL_TO_INDEX = {'cat': 0, 'dog': 1}
IMAGE_EXTENSIONS = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')
SEED = 1234

def label_from_path(path: Path) -> str:
    label = path.parent.name
    if label not in LABEL_TO_INDEX:
        raise ValueError(f'Unexpected label folder: {path}')
    return label

def load_preview_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB'))

def list_image_paths(label: str) -> list[Path]:
    label_dir = DATA_ROOT / label
    paths = []
    for pattern in IMAGE_EXTENSIONS:
        paths.extend(label_dir.glob(pattern))
    return sorted(paths)

def shuffled_paths(paths: list[Path], seed_offset: int=0) -> list[Path]:
    rng = np.random.default_rng(SEED + seed_offset)
    indices = rng.permutation(len(paths))
    return [paths[int(idx)] for idx in indices]

def sample_paths(paths: list[Path], count: int, seed_offset: int) -> list[Path]:
    ordered = shuffled_paths(paths, seed_offset=seed_offset)
    return ordered[:min(count, len(ordered))]

def sample_per_class(paths: list[Path], n_per_class: int, seed_offset: int=0) -> list[Path]:
    sampled = []
    for label_index, label in enumerate(LABELS):
        label_paths = [path for path in paths if label_from_path(path) == label]
        sampled.extend(sample_paths(label_paths, n_per_class, seed_offset + 50 * label_index))
    return sampled

def split_train_test(paths: list[Path], train_ratio: float=0.7, seed_offset: int=0):
    shuffled = shuffled_paths(paths, seed_offset)
    split_idx = int(len(shuffled) * train_ratio)
    return (shuffled[:split_idx], shuffled[split_idx:])

def load_image_np(path: Path) -> np.ndarray:
    raise NotImplementedError('Load one RGB image into a NumPy array.')

def center_crop(image: np.ndarray, crop_size: int=48) -> np.ndarray:
    raise NotImplementedError('Implement a centered crop with slicing.')

def flip_horizontal(image: np.ndarray) -> np.ndarray:
    raise NotImplementedError('Flip the image horizontally with slicing.')

def normalize_01(image: np.ndarray) -> np.ndarray:
    raise NotImplementedError('Normalize pixel values to [0, 1].')

def show_histograms(uint8_img, float_img):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(uint8_img.ravel(), bins=50)
    plt.title('Before (uint8: 0–255)')
    plt.subplot(1, 2, 2)
    plt.hist(float_img.ravel(), bins=50)
    plt.title('After (float: 0–1)')
    plt.tight_layout()
    plt.show()

def rgb_to_gray(image_float: np.ndarray) -> np.ndarray:
    raise NotImplementedError('Convert RGB to grayscale.')

def channel_summary(image_float: np.ndarray) -> tuple[np.ndarray, int]:
    raise NotImplementedError('Summarize the RGB channels with axis=(0, 1).')

def convolve2d_matmul(image_gray: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    raise NotImplementedError('Apply a 2D filter with matrix multiplication.')

def flatten_image(image: np.ndarray) -> np.ndarray:
    raise NotImplementedError('Flatten the image into one vector.')
FEATURE_NAMES = ['mean_r', 'mean_g', 'mean_b', 'std_r', 'std_g', 'std_b', 'brightest_channel', 'edge_mean', 'edge_std', 'row_std_mean']

def extract_features(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    cropped = center_crop(image, crop_size=48)
    image_float = normalize_01(cropped)
    gray = rgb_to_gray(image_float)
    channel_means, brightest_channel = channel_summary(image_float)
    channel_stds = image_float.std(axis=(0, 1)).astype(np.float32)
    filtered = convolve2d_matmul(gray, kernel)
    row_std_profile = np.apply_along_axis(np.std, 1, gray)
    raise NotImplementedError('Build one hand-crafted feature vector.')

def build_feature_matrix(paths: list[Path], kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError('Build the feature matrix for the dataset subset.')
