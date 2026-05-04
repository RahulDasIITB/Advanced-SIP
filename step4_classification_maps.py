"""
STEP 4: Generate Classification Maps
=====================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches


# Color palette for up to 20 classes
CLASS_COLORS = [
    '#000000',  # 0  - Background (black)
    '#FF0000',  # 1  - Red
    '#00FF00',  # 2  - Lime
    '#0000FF',  # 3  - Blue
    '#FFFF00',  # 4  - Yellow
    '#FF00FF',  # 5  - Magenta
    '#00FFFF',  # 6  - Cyan
    '#FF8000',  # 7  - Orange
    '#8000FF',  # 8  - Purple
    '#00FF80',  # 9  - Spring Green
    '#FF0080',  # 10 - Rose
    '#0080FF',  # 11 - Sky Blue
    '#80FF00',  # 12 - Yellow Green
    '#FF8080',  # 13 - Salmon
    '#80FF80',  # 14 - Light Green
    '#8080FF',  # 15 - Periwinkle
    '#FFFF80',  # 16 - Light Yellow
    '#FF80FF',  # 17 - Light Magenta
    '#80FFFF',  # 18 - Light Cyan
    '#C0C0C0',  # 19 - Silver
]

# Indian Pines class names
INDIAN_PINES_CLASSES = [
    'Background',
    'Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn',
    'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
    'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
    'Soybean-clean', 'Wheat', 'Woods',
    'Buildings-Grass-Trees-Drives', 'Stone-Steel-Towers'
]


def generate_classification_map(model, pixels, labels, shape, model_name="Model", pca=None, scaler=None):
    h, w, _ = shape

    print(f"[INFO] Generating classification map for {model_name}...")

    # 🔥 Apply correct pipeline
    if scaler is not None:
        pixels = scaler.transform(pixels)

    if pca is not None:
        pixels = pca.transform(pixels)

    pred_probs = model.predict(pixels, verbose=0)
    pred_all   = np.argmax(pred_probs, axis=1) + 1

    if labels is not None:
        pred_all[labels == 0] = 0

    pred_map = pred_all.reshape(h, w)

    print(f"[INFO] Map generated. Unique classes: {np.unique(pred_map)}")
    return pred_map, pred_all


def get_colormap(num_classes):
    colors = CLASS_COLORS[:num_classes + 1]
    cmap   = mcolors.ListedColormap(colors)
    bounds = np.arange(-0.5, num_classes + 1.5, 1)
    norm   = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def plot_classification_maps(gt, map_ga, map_random, num_classes,
                              class_names=None, has_gt=True):
    cmap, norm = get_colormap(num_classes)

    if has_gt and gt is not None:
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        maps   = [gt,       map_ga,              map_random]
        titles = ["Ground Truth Map", "GA Initialized NN", "Random Initialized NN"]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        maps   = [map_ga,           map_random]
        titles = ["GA Initialized NN", "Random Initialized NN"]

    for ax, m, title in zip(axes, maps, titles):
        ax.imshow(m, cmap=cmap, norm=norm)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.axis('off')

    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes + 1)]

    patches = []
    for i in range(num_classes + 1):
        if i < len(CLASS_COLORS):
            patch = mpatches.Patch(
                color=CLASS_COLORS[i],
                label=class_names[i] if i < len(class_names) else f"Class {i}"
            )
            patches.append(patch)

    fig.legend(handles=patches, loc='lower center', ncol=6,
               fontsize=9, bbox_to_anchor=(0.5, -0.02),
               frameon=True, fancybox=True)

    plt.suptitle("Satellite Image Classification Maps", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('classification_maps.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("[INFO] Classification maps saved as 'classification_maps.png'")


def plot_confidence_maps(model_ga, model_random, pixels, shape):
    h, w, _ = shape

    probs_ga     = model_ga.predict(pixels, verbose=0)
    probs_random = model_random.predict(pixels, verbose=0)

    conf_ga     = np.max(probs_ga,     axis=1).reshape(h, w)
    conf_random = np.max(probs_random, axis=1).reshape(h, w)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    im0 = axes[0].imshow(conf_ga,     cmap='RdYlGn', vmin=0, vmax=1)
    axes[0].set_title("GA Model — Confidence Map", fontsize=13, fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], label='Confidence')

    im1 = axes[1].imshow(conf_random, cmap='RdYlGn', vmin=0, vmax=1)
    axes[1].set_title("Random Model — Confidence Map", fontsize=13, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], label='Confidence')

    plt.suptitle("Model Confidence Maps\n(Green = Certain | Red = Uncertain)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('confidence_maps.png', dpi=150)
    plt.show()
    print("[INFO] Confidence maps saved as 'confidence_maps.png'")


def compute_mean_confidence(model_ga, model_random, pixels):
    probs_ga     = model_ga.predict(pixels, verbose=0)
    probs_random = model_random.predict(pixels, verbose=0)

    mean_conf_ga     = float(np.mean(np.max(probs_ga,     axis=1)))
    mean_conf_random = float(np.mean(np.max(probs_random, axis=1)))

    print(f"[INFO] GA Model Mean Confidence    : {mean_conf_ga:.4f}")
    print(f"[INFO] Random Model Mean Confidence: {mean_conf_random:.4f}")

    return {
        "ga_confidence"    : mean_conf_ga,
        "random_confidence": mean_conf_random
    }


if __name__ == "__main__":
    print("[INFO] This module is imported by other steps.")
    print("[INFO] Run app.py (Streamlit UI) for full pipeline.")