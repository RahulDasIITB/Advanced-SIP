import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import tempfile, os

from step1_load_data import load_data
from step2_neural_network import build_model, train_model
from step3_genetic_algorithm import run_genetic_algorithm, apply_ga_weights
from step4_classification_maps import generate_classification_map

st.set_page_config(page_title="Satellite Image Classifier", layout="wide")

st.title("Satellite Image Classification")
st.write("GA Initialization vs Random Initialization")

# ── Sidebar ─────────────────────────────────────────────
st.sidebar.header("Settings")
num_generations  = st.sidebar.slider("GA Generations", 10, 100, 30)
sol_per_pop      = st.sidebar.slider("GA Population", 10, 60, 20)
mutation_percent = st.sidebar.slider("Mutation %", 1, 20, 10)
epochs           = st.sidebar.slider("Training Epochs", 20, 200, 80)

# FAST DEMO MODE
use_precomputed = st.sidebar.checkbox("Fast Demo Mode")

# ── Upload ─────────────────────────────────────────────
st.subheader("Upload Files")
col1, col2 = st.columns(2)

with col1:
    image_file = st.file_uploader(
        "Upload Image",
        type=["mat","tif","tiff","png","jpg","jpeg","npy"]
    )

with col2:
    gt_file = st.file_uploader(
        "Upload Ground Truth (optional)",
        type=["mat","tif","png","jpg","npy"]
    )

# ── Run ─────────────────────────────────────────────
if st.button("Run Classification"):

    if image_file is None:
        st.error("Please upload an image.")
        st.stop()

    # Save uploaded files
    img_ext = os.path.splitext(image_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=img_ext) as f:
        f.write(image_file.read())
        img_path = f.name

    gt_path = None
    if gt_file:
        gt_ext = os.path.splitext(gt_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=gt_ext) as f:
            f.write(gt_file.read())
            gt_path = f.name

    # ── Load Data ─────────────────────────────────────
    data = load_data(img_path, gt_path)

    h, w, bands = data["shape"]
    st.success(f"Loaded: {h} x {w}, Bands = {bands}")
    st.info(f"Train: {len(data['X_train'])} | Test: {len(data['X_test'])} | Classes: {data['num_classes']}")

    if "pca" not in data or "scaler" not in data or "pixels_raw" not in data:
        st.error("Data loader missing required components.")
        st.stop()

    X_train     = data["X_train"]
    y_train     = data["y_train"]
    input_dim   = X_train.shape[1]
    num_classes = data["num_classes"]

    # ───────────────────────────────────────────────────
    # FAST DEMO MODE
    # ───────────────────────────────────────────────────
    if use_precomputed:

        st.info("Fast Demo Mode: Loading precomputed results")

        if not os.path.exists("map_ga.npy"):
            st.error("No precomputed files found. Run once without Fast Mode.")
            st.stop()

        map_ga   = np.load("map_ga.npy")
        map_rand = np.load("map_rand.npy")

        metrics = np.load("metrics.npy", allow_pickle=True).item()

        acc_ga_train   = metrics["acc_ga_train"]
        acc_rand_train = metrics["acc_rand_train"]
        acc_ga_val     = metrics["acc_ga_val"]
        acc_rand_val   = metrics["acc_rand_val"]
        test_acc_ga    = metrics["test_ga"]
        test_acc_rand  = metrics["test_rand"]

    # ───────────────────────────────────────────────────
    # FULL PIPELINE
    # ───────────────────────────────────────────────────
    else:

        # Build models
        model_ga     = build_model(input_dim, num_classes)
        model_random = build_model(input_dim, num_classes)

        # ── GA ─────────────────────────────────────
        st.subheader("Genetic Algorithm")

        progress_bar = st.progress(0)
        status_text  = st.empty()
        fitness_vals = []

        def ga_callback(gen, total, best_f):
            fitness_vals.append(best_f)
            progress_bar.progress(gen / total)
            status_text.text(f"Gen {gen}/{total} | Best Fitness: {max(fitness_vals):.4f}")

        best_solution, _, _ = run_genetic_algorithm(
            X_train, y_train, input_dim, num_classes,
            num_generations=num_generations,
            sol_per_pop=sol_per_pop,
            mutation_percent=mutation_percent,
            on_generation_callback=ga_callback
        )

        progress_bar.progress(1.0)

        true_best = max(fitness_vals)
        st.success(f"GA Completed | Best Fitness: {true_best:.4f}")

        # Plot GA curve
        fitness_curve = np.maximum.accumulate(fitness_vals)
        fig_ga, ax = plt.subplots()
        ax.plot(fitness_curve, marker='o')
        ax.set_title("GA Fitness (Best So Far)")
        st.pyplot(fig_ga)

        # Apply GA weights
        model_ga = apply_ga_weights(model_ga, best_solution)

        # ── Train models ─────────────────────────
        hist_ga   = train_model(model_ga,     X_train, y_train, epochs=epochs)
        hist_rand = train_model(model_random, X_train, y_train, epochs=epochs)

        # ── Training Curves ─────────────────────
        st.subheader("Training Curves")

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        axes[0,0].plot(hist_ga.history['accuracy'],   label="GA")
        axes[0,0].plot(hist_rand.history['accuracy'], label="Random")
        axes[0,0].set_title("Train Accuracy")
        axes[0,0].legend()

        axes[0,1].plot(hist_ga.history['val_accuracy'],   label="GA")
        axes[0,1].plot(hist_rand.history['val_accuracy'], label="Random")
        axes[0,1].set_title("Val Accuracy")
        axes[0,1].legend()

        axes[1,0].plot(hist_ga.history['loss'],   label="GA")
        axes[1,0].plot(hist_rand.history['loss'], label="Random")
        axes[1,0].set_title("Train Loss")
        axes[1,0].legend()

        axes[1,1].plot(hist_ga.history['val_loss'],   label="GA")
        axes[1,1].plot(hist_rand.history['val_loss'], label="Random")
        axes[1,1].set_title("Val Loss")
        axes[1,1].legend()

        st.pyplot(fig)

        # ── Accuracy ─────────────────────────────
        acc_ga_train   = hist_ga.history["accuracy"][-1]
        acc_rand_train = hist_rand.history["accuracy"][-1]
        acc_ga_val     = hist_ga.history["val_accuracy"][-1]
        acc_rand_val   = hist_rand.history["val_accuracy"][-1]

        # ── Maps ────────────────────────────────
        map_ga, _ = generate_classification_map(
            model_ga, data["pixels_raw"], data["labels"], data["shape"],
            pca=data["pca"], scaler=data["scaler"]
        )

        map_rand, _ = generate_classification_map(
            model_random, data["pixels_raw"], data["labels"], data["shape"],
            pca=data["pca"], scaler=data["scaler"]
        )

        # ── Test Accuracy ───────────────────────
        from sklearn.metrics import accuracy_score

        y_pred_ga   = np.argmax(model_ga.predict(data["X_test"]),     axis=1)
        y_pred_rand = np.argmax(model_random.predict(data["X_test"]), axis=1)

        test_acc_ga   = accuracy_score(data["y_test"], y_pred_ga)
        test_acc_rand = accuracy_score(data["y_test"], y_pred_rand)

        # SAVE FOR DEMO
        np.save("map_ga.npy",   map_ga)
        np.save("map_rand.npy", map_rand)

        np.save("metrics.npy", {
            "acc_ga_train"  : acc_ga_train,
            "acc_rand_train": acc_rand_train,
            "acc_ga_val"    : acc_ga_val,
            "acc_rand_val"  : acc_rand_val,
            "test_ga"       : test_acc_ga,
            "test_rand"     : test_acc_rand
        })

    # ── Display Results ───────────────────────────────
    st.subheader("Accuracy Summary")

    c1, c2 = st.columns(2)
    c1.metric("GA Train",     f"{acc_ga_train:.3f}")
    c2.metric("Random Train", f"{acc_rand_train:.3f}")

    c3, c4 = st.columns(2)
    c3.metric("GA Val",     f"{acc_ga_val:.3f}")
    c4.metric("Random Val", f"{acc_rand_val:.3f}")

    st.subheader("Test Accuracy")
    t1, t2 = st.columns(2)
    t1.metric("GA",     f"{test_acc_ga:.3f}")
    t2.metric("Random", f"{test_acc_rand:.3f}")

    # ── Classification Maps ───────────────────────────
    st.subheader("Classification Maps")

    # Indian Pines class names
    INDIAN_PINES_CLASSES = [
        'Background',
        'Alfalfa',
        'Corn-notill',
        'Corn-mintill',
        'Corn',
        'Grass-pasture',
        'Grass-trees',
        'Grass-pasture-mowed',
        'Hay-windrowed',
        'Oats',
        'Soybean-notill',
        'Soybean-mintill',
        'Soybean-clean',
        'Wheat',
        'Woods',
        'Buildings-Grass-Trees-Drives',
        'Stone-Steel-Towers'
    ]

    # Colors — index 0 = background, 1..16 = classes
    ALL_COLORS = [
        '#3B0764',  # 0  Background
        '#FF0000',  # 1  Alfalfa
        '#00CC00',  # 2  Corn-notill
        '#0000FF',  # 3  Corn-mintill
        '#FFFF00',  # 4  Corn
        '#FF00FF',  # 5  Grass-pasture
        '#00FFFF',  # 6  Grass-trees
        '#FF8000',  # 7  Grass-pasture-mowed
        '#8000FF',  # 8  Hay-windrowed
        '#00FF80',  # 9  Oats
        '#FF0080',  # 10 Soybean-notill
        '#0080FF',  # 11 Soybean-mintill
        '#80FF00',  # 12 Soybean-clean
        '#FF8080',  # 13 Wheat
        '#80FF80',  # 14 Woods
        '#8080FF',  # 15 Buildings-Grass-Trees-Drives
        '#FFFF80',  # 16 Stone-Steel-Towers
    ]

    colors = ALL_COLORS[:num_classes + 1]
    cmap   = mcolors.ListedColormap(colors)
    bounds = np.arange(-0.5, num_classes + 1.5, 1)
    norm   = mcolors.BoundaryNorm(bounds, cmap.N)

    # ── Plot maps ─────────────────────────────────────
    if data["gt"] is not None:
        fig_maps, ax = plt.subplots(1, 3, figsize=(18, 6))

        ax[0].imshow(data["gt"], cmap=cmap, norm=norm)
        ax[0].set_title("Ground Truth", fontsize=13, fontweight='bold')
        ax[0].axis("off")

        ax[1].imshow(map_ga, cmap=cmap, norm=norm)
        ax[1].set_title("GA Model", fontsize=13, fontweight='bold')
        ax[1].axis("off")

        ax[2].imshow(map_rand, cmap=cmap, norm=norm)
        ax[2].set_title("Random Model", fontsize=13, fontweight='bold')
        ax[2].axis("off")

    else:
        fig_maps, ax = plt.subplots(1, 2, figsize=(12, 6))

        ax[0].imshow(map_ga,   cmap=cmap, norm=norm)
        ax[0].set_title("GA Model", fontsize=13, fontweight='bold')
        ax[0].axis("off")

        ax[1].imshow(map_rand, cmap=cmap, norm=norm)
        ax[1].set_title("Random Model", fontsize=13, fontweight='bold')
        ax[1].axis("off")

    plt.tight_layout()
    st.pyplot(fig_maps)

    #  Class Legend ───────────────────────────
    st.markdown("####Class Legend")

    legend_items = [
        (i, INDIAN_PINES_CLASSES[i], colors[i])
        for i in range(num_classes + 1)
        if i < len(colors) and i < len(INDIAN_PINES_CLASSES)
    ]

    items_per_row = 4
    for row_start in range(0, len(legend_items), items_per_row):
        row_items = legend_items[row_start : row_start + items_per_row]
        cols      = st.columns(items_per_row)

        for col, (class_idx, class_name, color) in zip(cols, row_items):
            col.markdown(
                f"""
                <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
                    <div style="width:22px; height:22px; background:{color};
                                border-radius:4px; border:1px solid #555;
                                flex-shrink:0;"></div>
                    <span style="font-size:13px;">
                        <b>{class_idx}</b> — {class_name}
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Cleanup
    os.remove(img_path)
    if gt_path:
        os.remove(gt_path)