"""
main.py — Master Pipeline
==========================
Runs the complete satellite image classification pipeline:

    Step 1 → Load & Preprocess Data
    Step 2 → Build Neural Networks
    Step 3 → Genetic Algorithm Weight Initialization
    Step 4 → Train Both Models
    Step 5 → Generate Classification Maps
    Step 6 → Accuracy Assessment

Usage:
    python main.py

To launch the Streamlit UI instead:
    streamlit run app.py

FIXES APPLIED:
  - test_size: 0.7 → 0.2  (was training on only 30% of data)
  - NUM_GENERATIONS: 20 → 50
  - SOL_PER_POP: 10 → 30
  - MUTATION_PERCENT: 10 → 5
  - EPOCHS: 50 → 100
"""

import numpy as np

from step1_load_data         import load_data, generate_pseudo_labels
from step2_neural_network    import (build_model, get_total_weights,
                                     train_model, plot_training_curves)
from step3_genetic_algorithm import (run_genetic_algorithm, apply_ga_weights,
                                     plot_ga_fitness)
from step4_classification_maps import (generate_classification_map,
                                        plot_classification_maps,
                                        plot_confidence_maps,
                                        compute_mean_confidence)
from step5_accuracy_assessment import (compute_clustering_metrics,
                                        compute_map_agreement,
                                        compute_supervised_metrics,
                                        plot_metrics_comparison,
                                        plot_class_distribution,
                                        plot_confusion_matrices)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
IMAGE_PATH   = r"D:\Documents\M.Tech GNR\2nd Sem\GNR-602 ASIP\SIP_Project\Model2\Data\Corrected\Indian_pines_corrected.mat"
GT_PATH      = r"D:\Documents\M.Tech GNR\2nd Sem\GNR-602 ASIP\SIP_Project\Model2\Data\Ground Truth\Indian_pines_gt.mat"        # Set to None if unavailable

# GA Settings — ✅ FIXED values
NUM_GENERATIONS  = 50     # was 20
SOL_PER_POP      = 30     # was 10
MUTATION_PERCENT = 5      # was 10

# Training Settings — ✅ FIXED
EPOCHS      = 100         # was 50
BATCH_SIZE  = 32

NUM_CLASSES_MANUAL = 5    # Only used when GT_PATH = None


# ─────────────────────────────────────────────────────────────────────────────
def main():
# ─────────────────────────────────────────────────────────────────────────────

    print("=" * 65)
    print("   SATELLITE IMAGE CLASSIFICATION PIPELINE")
    print("   GA Initialized NN  vs  Random Initialized NN")
    print("=" * 65)

    # ── STEP 1: Load Data ─────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("STEP 1: Loading and Preprocessing Data")
    print("─" * 65)

    data = load_data(IMAGE_PATH, GT_PATH)

    if not data["has_gt"]:
        print(f"[INFO] No ground truth. Generating {NUM_CLASSES_MANUAL} pseudo labels via KMeans...")
        y_pseudo = generate_pseudo_labels(data["pixels"], NUM_CLASSES_MANUAL)

        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(
            data["pixels"], y_pseudo,
            test_size=0.2,          # ✅ FIX: was 0.7
            random_state=42
        )
        data.update({
            "X"          : data["pixels"],
            "y"          : y_pseudo,
            "labels"     : y_pseudo,
            "X_train"    : X_tr,
            "X_test"     : X_te,
            "y_train"    : y_tr,
            "y_test"     : y_te,
            "num_classes": NUM_CLASSES_MANUAL
        })

    input_dim   = data["X_train"].shape[1]
    num_classes = data["num_classes"]

    print(f"\n[STEP 1 DONE]")
    print(f"  Image shape  : {data['shape']}")
    print(f"  Train samples: {len(data['X_train'])}  (80% of labeled pixels)")
    print(f"  Test samples : {len(data['X_test'])}   (20% of labeled pixels)")
    print(f"  Num classes  : {num_classes}")

    # ── STEP 2: Build Models ──────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("STEP 2: Building Neural Network")
    print("─" * 65)

    model_ga      = build_model(input_dim, num_classes)
    model_random  = build_model(input_dim, num_classes)
    total_weights = get_total_weights(model_ga)

    print(f"\n[STEP 2 DONE]")
    print(f"  Architecture : {input_dim} → 128 → 64 → 32 → {num_classes}")
    print(f"  Total weights: {total_weights}")

    # ── STEP 3: Genetic Algorithm ─────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("STEP 3: Running Genetic Algorithm")
    print("─" * 65)

    best_solution, best_fitness, ga_instance = run_genetic_algorithm(
        data["X_train"], data["y_train"],
        input_dim, num_classes,
        num_generations  = NUM_GENERATIONS,
        sol_per_pop      = SOL_PER_POP,
        mutation_percent = MUTATION_PERCENT
    )

    model_ga = apply_ga_weights(model_ga, best_solution)
    plot_ga_fitness(ga_instance)

    print(f"\n[STEP 3 DONE]")
    print(f"  Best GA fitness: {best_fitness*100:.2f}%  (after 5 warmup epochs)")

    # ── STEP 4: Train Both Models ─────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("STEP 4: Training Both Models")
    print("─" * 65)

    history_ga = train_model(
        model_ga, data["X_train"], data["y_train"],
        epochs=EPOCHS, batch_size=BATCH_SIZE, label="GA Model"
    )

    history_random = train_model(
        model_random, data["X_train"], data["y_train"],
        epochs=EPOCHS, batch_size=BATCH_SIZE, label="Random Model"
    )

    plot_training_curves(history_ga, history_random)

    print(f"\n[STEP 4 DONE]")
    print(f"  GA Model     final train acc: {history_ga.history['accuracy'][-1]*100:.2f}%")
    print(f"  Random Model final train acc: {history_random.history['accuracy'][-1]*100:.2f}%")
    print(f"  GA Model     final val acc  : {history_ga.history['val_accuracy'][-1]*100:.2f}%")
    print(f"  Random Model final val acc  : {history_random.history['val_accuracy'][-1]*100:.2f}%")

    # ── STEP 5: Classification Maps ───────────────────────────────────────────
    print("\n" + "─" * 65)
    print("STEP 5: Generating Classification Maps")
    print("─" * 65)

    map_ga,     pred_ga_all     = generate_classification_map(
        model_ga,     data["pixels"], data["labels"], data["shape"], "GA Model"
    )
    map_random, pred_random_all = generate_classification_map(
        model_random, data["pixels"], data["labels"], data["shape"], "Random Model"
    )

    plot_classification_maps(
        gt          = data["gt"],
        map_ga      = map_ga,
        map_random  = map_random,
        num_classes = num_classes,
        has_gt      = data["has_gt"]
    )

    plot_confidence_maps(model_ga, model_random, data["pixels"], data["shape"])
    confidence = compute_mean_confidence(model_ga, model_random, data["pixels"])

    print(f"\n[STEP 5 DONE]")
    print(f"  GA Model Mean Confidence    : {confidence['ga_confidence']:.4f}")
    print(f"  Random Model Mean Confidence: {confidence['random_confidence']:.4f}")

    # ── STEP 6: Accuracy Assessment ───────────────────────────────────────────
    print("\n" + "─" * 65)
    print("STEP 6: Accuracy Assessment")
    print("─" * 65)

    sup_metrics = None
    if data["has_gt"]:
        print("\n[INFO] Ground truth available → Computing supervised accuracy...")
        y_pred_ga     = np.argmax(model_ga.predict(data["X_test"],     verbose=0), axis=1)
        y_pred_random = np.argmax(model_random.predict(data["X_test"], verbose=0), axis=1)

        sup_metrics = compute_supervised_metrics(
            data["y_test"], y_pred_ga, y_pred_random
        )
        plot_confusion_matrices(
            data["y_test"], y_pred_ga, y_pred_random, num_classes
        )
    else:
        print("\n[INFO] No ground truth → Skipping supervised accuracy.")

    print("\n[INFO] Computing internal clustering metrics...")
    clust_metrics = compute_clustering_metrics(
        data["pixels"], pred_ga_all, pred_random_all,
        sample_size=2000
    )

    plot_metrics_comparison(clust_metrics)
    agreement = compute_map_agreement(map_ga, map_random)
    plot_class_distribution(map_ga, map_random, num_classes)

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("FINAL SUMMARY")
    print("=" * 65)

    if data["has_gt"]:
        print(f"\n  Supervised Accuracy:")
        print(f"    GA Model     : {sup_metrics['ga_accuracy']*100:.2f}%")
        print(f"    Random Model : {sup_metrics['random_accuracy']*100:.2f}%")

    print(f"\n  Internal Clustering Metrics:")
    print(f"    {'Metric':<30} {'GA':>10} {'Random':>10} {'Better':>10}")
    print(f"    {'-'*62}")

    sil_ga = clust_metrics["ga"]["silhouette"]
    sil_rn = clust_metrics["random"]["silhouette"]
    db_ga  = clust_metrics["ga"]["davies_bouldin"]
    db_rn  = clust_metrics["random"]["davies_bouldin"]
    ch_ga  = clust_metrics["ga"]["calinski_harabasz"]
    ch_rn  = clust_metrics["random"]["calinski_harabasz"]

    print(f"    {'Silhouette Score (↑)':<30} {sil_ga:>10.4f} {sil_rn:>10.4f} {'GA ✅' if sil_ga > sil_rn else 'Random ✅':>10}")
    print(f"    {'Davies-Bouldin (↓)':<30} {db_ga:>10.4f} {db_rn:>10.4f}  {'GA ✅' if db_ga < db_rn else 'Random ✅':>10}")
    print(f"    {'Calinski-Harabasz (↑)':<30} {ch_ga:>10.2f} {ch_rn:>10.2f}  {'GA ✅' if ch_ga > ch_rn else 'Random ✅':>10}")

    print(f"\n  Confidence Scores:")
    print(f"    GA Model     : {confidence['ga_confidence']:.4f}")
    print(f"    Random Model : {confidence['random_confidence']:.4f}")
    print(f"\n  Map Agreement : {agreement:.2f}% pixels classified the same by both models")

    ga_wins = sum([
        sil_ga > sil_rn,
        db_ga  < db_rn,
        ch_ga  > ch_rn,
        confidence["ga_confidence"] > confidence["random_confidence"]
    ])
    if data["has_gt"]:
        ga_wins += int(sup_metrics["ga_accuracy"] > sup_metrics["random_accuracy"])

    total_metrics = 5 if data["has_gt"] else 4
    print(f"\n  GA wins {ga_wins} out of {total_metrics} metrics.")

    if ga_wins >= (total_metrics // 2 + 1):
        print("\n CONCLUSION: GA Initialized NN performs BETTER overall.")
    elif ga_wins == total_metrics // 2:
        print("\n CONCLUSION: Results are MIXED. Both models comparable.")
    else:
        print("\n CONCLUSION: Random Initialized NN performs comparably or better.")

    print("\n" + "=" * 65)
    print("Pipeline complete! Check saved plots in your project folder.")
    print("=" * 65)


if __name__ == "__main__":
    main()