"""
STEP 1: Load and Preprocess Satellite Image Data
=================================================
- Loads satellite image and optional ground truth from .mat files
- Normalizes pixel values
- Splits into train/test sets
 
"""
 
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
 
 
def load_data(image_path, gt_path=None):
    import os
    ext = os.path.splitext(image_path)[1].lower()

    # ── Load Image ────────────────────────────────────
    if ext == ".mat":
        img_data = sio.loadmat(image_path)
        img_key  = [k for k in img_data if not k.startswith("_")][0]
        image    = img_data[img_key].astype(np.float32)

    elif ext in [".png", ".jpg", ".jpeg"]:
        from PIL import Image
        image = np.array(Image.open(image_path).convert("RGB")).astype(np.float32)

    elif ext in [".tif", ".tiff"]:
        import rasterio
        with rasterio.open(image_path) as src:
            image = src.read()
            image = np.transpose(image, (1, 2, 0)).astype(np.float32)

    elif ext == ".npy":
        image = np.load(image_path).astype(np.float32)
        if image.ndim == 2:
            image = image[:, :, np.newaxis]

    else:
        raise ValueError(f"Unsupported file format: {ext}")

    if image.ndim == 2:
        image = image[:, :, np.newaxis]

    h, w, bands = image.shape
    print(f"[INFO] Image loaded: {h}x{w}, {bands} bands")

    # ── RAW PIXELS────────────────────────
    pixels_raw = image.reshape(-1, bands)

    # ── Scaling ──────────────────────────────────────
    scaler = MinMaxScaler()
    pixels_scaled = scaler.fit_transform(pixels_raw)

    # ── PCA ──────────────────────────────────────────
    from sklearn.decomposition import PCA
    pca = PCA(n_components=30)
    pixels_pca = pca.fit_transform(pixels_scaled)

    print(f"[INFO] PCA applied: {bands} → 30 components")

    # ── Result dictionary ─────────────────────────────
    result = {
        "image"      : image,
        "pixels_raw" : pixels_raw,    
        "pixels"     : pixels_pca,   
        "scaler"     : scaler,        
        "pca"        : pca,           
        "shape"      : (h, w, bands),
        "gt"         : None,
        "labels"     : None,
        "X"          : None,
        "y"          : None,
        "X_train"    : None,
        "X_test"     : None,
        "y_train"    : None,
        "y_test"     : None,
        "num_classes": None,
        "has_gt"     : False
    }

    # ── Load Ground Truth ─────────────────────────────
    if gt_path:
        gt_ext = os.path.splitext(gt_path)[1].lower()

        if gt_ext == ".mat":
            gt_data = sio.loadmat(gt_path)
            gt_key  = [k for k in gt_data if not k.startswith("_")][0]
            gt      = gt_data[gt_key]

        elif gt_ext in [".png", ".jpg", ".jpeg"]:
            from PIL import Image
            gt = np.array(Image.open(gt_path).convert("L"))

        elif gt_ext in [".tif", ".tiff"]:
            import rasterio
            with rasterio.open(gt_path) as src:
                gt = src.read(1)

        elif gt_ext == ".npy":
            gt = np.load(gt_path)

        labels = gt.reshape(-1)
        mask   = labels > 0

        X = pixels_pca[mask]
        y = labels[mask] - 1

        num_classes = len(np.unique(y))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"[INFO] Train: {len(X_train)} | Test: {len(X_test)}")

        result.update({
            "gt"         : gt,
            "labels"     : labels,
            "X"          : X,
            "y"          : y,
            "X_train"    : X_train,
            "X_test"     : X_test,
            "y_train"    : y_train,
            "y_test"     : y_test,
            "num_classes": num_classes,
            "has_gt"     : True
        })

    return result
 
 
def generate_pseudo_labels(pixels, num_classes):
    """
    When no ground truth is available, use KMeans to generate pseudo labels.
    """
    from sklearn.cluster import KMeans
 
    print(f"[INFO] Generating pseudo labels using KMeans (k={num_classes})...")
    kmeans = KMeans(n_clusters=num_classes, random_state=42, n_init=10)
    y = kmeans.fit_predict(pixels)
    print(f"[INFO] Pseudo labels generated.")
    return y
 
 
def visualize_ground_truth(gt, title="Ground Truth Map"):
    plt.figure(figsize=(6, 6))
    plt.imshow(gt, cmap='jet')
    plt.colorbar(label='Class Label')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('ground_truth_map.png', dpi=150)
    plt.show()
    print("[INFO] Ground truth map saved as 'ground_truth_map.png'")
 
 
if __name__ == "__main__":
    IMAGE_PATH = r"D:\Documents\M.Tech GNR\2nd Sem\GNR-602 ASIP\SIP_Project\Model2\Data\Corrected\Indian_pines_corrected.mat"
    GT_PATH    = r"D:\Documents\M.Tech GNR\2nd Sem\GNR-602 ASIP\SIP_Project\Model2\Data\Ground Truth\Indian_pines_gt.mat"
 
    data = load_data(IMAGE_PATH, GT_PATH)
 
    if data["has_gt"]:
        visualize_ground_truth(data["gt"])
        print(f"\n[RESULT] Data ready.")
        print(f"  X_train: {data['X_train'].shape}")
        print(f"  X_test : {data['X_test'].shape}")
        print(f"  Classes: {data['num_classes']}")