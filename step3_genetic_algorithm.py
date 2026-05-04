"""
STEP 3: Genetic Algorithm Weight Initialization
================================================
"""

import gc
import numpy as np
import pygad
import matplotlib.pyplot as plt
from step2_neural_network import build_model, set_weights_from_chromosome, get_total_weights


def run_genetic_algorithm(X_train, y_train, input_dim, num_classes,
                           num_generations=50,
                           sol_per_pop=30,
                           num_parents_mating=6,
                           mutation_percent=5,
                           on_generation_callback=None):
    """
    Run Genetic Algorithm to find optimal initial weights for the NN.

    GA Workflow:
        1. Initialize population using He-scale range
        2. Each chromosome = NN weights, trained for 10 warmup epochs
        3. Fitness = accuracy after warmup on 500 sample subset
        4. Select top chromosomes, crossover, mutate
        5. Repeat for num_generations
        6. Return GLOBAL best chromosome (best across ALL generations)

    """

    # ── Get chromosome length from model ─────────────────────────────────────
    temp_model    = build_model(input_dim, num_classes)
    total_weights = get_total_weights(temp_model)
    del temp_model
    gc.collect()

    # He initialization scale
    he_std = float(np.sqrt(2.0 / input_dim))

    print(f"\n[GA] Starting Genetic Algorithm")
    print(f"     Generations      : {num_generations}")
    print(f"     Population size  : {sol_per_pop}")
    print(f"     Chromosome length: {total_weights}  (was ~36,592 — now much smaller!)")
    print(f"     Mutation %       : {mutation_percent}%")
    print(f"     Init range       : [{-he_std:.4f}, {he_std:.4f}]  (He scale)")
    print(f"     Warmup epochs    : 10  (increased from 5 for better fitness signal)")

    #  random subset for fitness evaluation ────────────────────────────
    SAMPLE_SIZE = min(500, len(X_train))
    sample_idx  = np.random.choice(len(X_train), SAMPLE_SIZE, replace=False)
    X_sample    = X_train[sample_idx]
    y_sample    = y_train[sample_idx]

    print(f"     Fitness sample   : {SAMPLE_SIZE} samples")

    # ── Global best tracker ───────────────────────────────────────────────────
    global_best_fitness  = -np.inf
    global_best_solution = None

    # ── Fitness Function ──────────────────────────────────────────────────────
    def fitness_func(ga_instance, solution, solution_idx):
        """
        Fitness = accuracy after 10 warmup epochs on 500 samples.

        CHANGES:
          - epochs increased 5 → 10 (clearer fitness signal)
          - gc.collect() instead of clear_session() (safe memory free)
        """
        model = build_model(input_dim, num_classes)
        model = set_weights_from_chromosome(model, solution)


        _, acc = model.evaluate(X_sample, y_sample, verbose=0)

        del model
        gc.collect()

        return float(acc)

    # ── Generation Callback ───────────────────────────────────────────────────
    def on_generation(ga_instance):
        nonlocal global_best_fitness, global_best_solution

        gen                  = ga_instance.generations_completed
        solution, best_f, _  = ga_instance.best_solution()

        if best_f > global_best_fitness:
            global_best_fitness  = best_f
            global_best_solution = solution.copy()
            print(f"  [GA] Gen {gen:3d}/{num_generations} | "
                  f"Fitness: {best_f:.4f}  ✅ New global best!")
        else:
            print(f"  [GA] Gen {gen:3d}/{num_generations} | "
                  f"Fitness: {best_f:.4f}  (global best: {global_best_fitness:.4f})")

        if on_generation_callback:
            on_generation_callback(gen, num_generations, global_best_fitness)

    # ── GA Instance ───────────────────────────────────────────────────────────
    ga_instance = pygad.GA(
        num_generations         = num_generations,
        num_parents_mating      = num_parents_mating,
        fitness_func            = fitness_func,
        sol_per_pop             = sol_per_pop,
        num_genes               = int(total_weights),

        # He scale initialization
        init_range_low          = -he_std,
        init_range_high         =  he_std,

        parent_selection_type   = "tournament",
        crossover_type          = "single_point",
        mutation_type           = "random",

        random_mutation_min_val = -0.1,
        random_mutation_max_val =  0.1,

        mutation_percent_genes  = mutation_percent,
        keep_elitism            = 2,
        on_generation           = on_generation
    )

    # ── Run GA ────────────────────────────────────────────────────────────────
    ga_instance.run()

    print(f"\n[GA] Complete!")
    print(f"[GA] Best fitness (acc after 10 warmup epochs): {global_best_fitness:.4f}")
    print(f"[GA] Global best tracked across all {num_generations} generations")

    return global_best_solution, global_best_fitness, ga_instance


def apply_ga_weights(model, best_solution):
    """Apply the best GA chromosome as the model's initial weights."""
    model = set_weights_from_chromosome(model, best_solution)
    print("[INFO] GA weights applied to model.")
    return model


def plot_ga_fitness(ga_instance):
    plt.close('all')
    plt.figure(figsize=(8, 4))

    raw_fitness = ga_instance.best_solutions_fitness
    monotonic   = np.maximum.accumulate(raw_fitness)

    plt.plot(monotonic,   linewidth=2, color='green',      label='Global Best Fitness')
    plt.plot(raw_fitness, linewidth=1, color='lightgreen',
             linestyle='--', alpha=0.6,                    label='Per-gen Best')
    plt.title("GA Fitness Over Generations (Best So Far)")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (Accuracy after warmup)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ga_fitness.png', dpi=150)
    plt.show()
    print("[INFO] GA fitness plot saved as 'ga_fitness.png'")


if __name__ == "__main__":
    import scipy.io as sio
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split

    IMAGE_PATH = r"D:\Documents\M.Tech GNR\2nd Sem\GNR-602 ASIP\SIP_Project\Model2\Data\Corrected\Indian_pines_corrected.mat"
    GT_PATH    = r"D:\Documents\M.Tech GNR\2nd Sem\GNR-602 ASIP\SIP_Project\Model2\Data\Ground Truth\Indian_pines_gt.mat"

    img_data = sio.loadmat(IMAGE_PATH)
    gt_data  = sio.loadmat(GT_PATH)

    image = img_data['indian_pines_corrected'].astype(np.float32)
    gt    = gt_data['indian_pines_gt']

    h, w, bands = image.shape
    pixels  = image.reshape(-1, bands)
    labels  = gt.reshape(-1)

    scaler  = MinMaxScaler()
    pixels  = scaler.fit_transform(pixels)

    # PCA reduction to 30 components
    pca    = PCA(n_components=30)
    pixels = pca.fit_transform(pixels)
    print(f"[INFO] PCA: 200 bands → 30 components "
          f"({pca.explained_variance_ratio_.sum()*100:.1f}% variance retained)")

    mask = labels > 0
    X    = pixels[mask]
    y    = labels[mask] - 1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    num_classes = len(np.unique(y))
    input_dim   = X_train.shape[1]   # = 30 after PCA

    best_solution, best_fitness, ga_instance = run_genetic_algorithm(
        X_train, y_train, input_dim, num_classes
    )

    plot_ga_fitness(ga_instance)