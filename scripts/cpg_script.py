import argparse
import os
from functools import partial
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
import jax.numpy as jnp
import numpy as np
from data.tables import codon_to_aa
from pathlib import Path

nuc_to_int = {"A":0,"C":1, "G":2, "T":3}


def check_cds(text: str) -> str:
    cds = "".join(i for i in text.upper() if i in "ACGT")
    if len(cds) % 3 != 0:
        raise ValueError(f"Sequence CDS length must be divisible by 3, got length {len(cds)}.")
    return cds

def read_sequence_cds(path):
    with open(path, "r", encoding="utf-8") as f:
        return check_cds(f.read())


def codon_to_nt_ints(codon: str):
    return [nuc_to_int[i] for i in codon]


def build_synonymous_tables(cds: str):
    """Returns:
    codon_choices_nt """

    codons = [cds[i:i+3] for i in range(0, len(cds), 3)]
    aa_seq = [codon_to_aa[i] for i in codons]

    ### might remove this since having an internal stop codon could indicate no fitness and be useful
    if "*" in aa_seq[:-1]:
        raise ValueError("Internal stop codon found in amino acid sequence")
    
    aa_to_codons = {}
    for codon, aa in codon_to_aa.items():
        aa_to_codons.setdefault(aa, []).append(codon) # create dict aa --> codon
    
    #if terminal stop codon exist, lets keep synonymous options
    choices_per_position = [aa_to_codons[aa] for aa in aa_seq]
    max_syn = max(len(i) for i in choices_per_position)
    n_codons = len(codons) # number of codons sequence contains

    codon_choices_nt = np.zeros((n_codons, max_syn, 3), dtype=np.uint8)
    syn_counts = np.zeros((n_codons,), dtype=np.uint8) # 1D array to act as a count for each position
    wt_choice_indices = np.zeros((n_codons,), dtype=np.uint8)

    for i, choices in enumerate(choices_per_position):
        syn_counts[i] = len(choices) # update syn_counts with the number of codons, for the aa, available in that position

        for j, codon in enumerate(choices):
            codon_choices_nt[i, j, :] = codon_to_nt_ints(codon) # each codon is broken into ints as referenced in codon_to_nt_ints
        
        wt_codon = codons[i] # define wt codons
        wt_choice_indices[i] = choices.index(wt_codon) # save index of wt codon
    return codon_choices_nt, syn_counts, wt_choice_indices



@jax.jit
def population_to_nt(population, codon_choices_nt):
    """
    codon_choices_nt: (codon pos, synonymous choices per codon, nucleotides per codon) e.g. (2,3,3)
    population: (candidate sequences, codon positions per sequence) e.g (3,2)

    codon_choices_nt example:

    position 0: 3 codon choaices [0,0,0] and [1,1,1] etc..
    [[0, 0, 0],
     [1, 1, 1],
     [2, 2, 2]],

    position 1: 3 codon choices [0,1,0] and [1,2,1] adn so on 0,1,0 = ACA
    [[0, 1, 0],
     [1, 2, 1],
     [2, 3, 2]],

    so codon_choices_nt indicates the allowed codons at each position

    population: example:
    3 sequences, 2 codon positions per sequence - each row is one sequence.
    Each integer indicates which codon choice to use at that positon
    [0, 0],
    [1, 2],
    [2, 1],

    since sequence 0 = [0,0]: at position 0, choose choice 0: AAA
    at position 1, choose choice 0: ACA

    """
    n_codons = population.shape[1]
    positions = jnp.arange(n_codons)[None, :]
    nt = codon_choices_nt[positions, population]

    return nt.reshape(population.shape[0], n_codons * 3)



def cpg_counts_from_population(population, codon_choices_nt):

    """map the CpG sites in each sequence """

    seq = population_to_nt(population, codon_choices_nt) # get sequence from above function
    cytosine_mask = seq[:, :-1] == 1 # mask for places where 1 (C)
    guanine_mask = seq[:, 1:] == 2 # mask for palces where 2 (G)

    cpg_sites = cytosine_mask & guanine_mask # look for places with combined mask C --> G


    return jnp.sum(cpg_sites, axis=1)


@jax.jit
def gc_counts_from_population(population, codon_choices_nt):

    seq = population_to_nt(population, codon_choices_nt)
    cytosine_mask = seq == 1
    guanine_mask = seq == 2

    gc_sites = cytosine_mask | guanine_mask

    return jnp.sum(gc_sites, axis=1)


@partial(jax.jit, static_argnames=("target_metric",))
def metric_counts_from_population(population, codon_choices_nt, target_metric):

    """
    Define which metric the EA will use

    target_matric:(str): cpg or gc

    """

    if target_metric == "cpg":
        return cpg_counts_from_population(population, codon_choices_nt)
    elif target_metric == "gc":
        return gc_counts_from_population(population, codon_choices_nt)
    else:
        raise ValueError("target_metric must be 'cpg' or 'gc'")




@partial(jax.jit, static_argnames=("population_size",))
def initialize_population(key, population_size, syn_counts):
    """ 
    key (int): JAX random number generator state
    population_size (int): number of sequences in the population
    syn_counts: 1D array defining the number of synonymous choices at each position of the sequence - derived from build_synonymous_tables()
    e.g. syn_counts = [1, 6, 2, 4]: position 1 has 1 synonymous choice, 
    position 2 has 6 synonymous choices,
    posiiton 3 has 2 synonymous choices and so on...

    population:
    outputs choice indicies:
    [[0 4 1 2]
    [0 3 0 3]
    [0 4 0 3]
    [0 5 0 3]]

    where for sequence 1: [0 4 1 2], posiiton 0 chose synonymous codon option 0,
    posiiton 1 chose synonymous codon option 4,
    position 2 chose synonymous codon option 1 and so on for each sequence. 


    """

    n_codons = syn_counts.shape[0]
    random_float = jax.random.uniform(
        key,
        shape=(population_size, n_codons),
        minval=0.0,
        maxval=1.0
    )

    population = jnp.floor(random_float * syn_counts[None, :]).astype(jnp.uint8)


    return population




@partial(jax.jit, static_argnames=("elite_count", "target_metric"))
def evolve_step_metric_bin(
    key,
    population,
    codon_choices_nt,
    syn_counts,
    mutation_rate,
    elite_count,
    bin_low,
    bin_high,
    target_metric):
    """
    Simple Evolutionary Algorithm
    1. score the population
    2. keep elite sequences
    3. sample parents from elites
    4. mutate synonymous codon choices

    ######args########

    key (JAX array / PRNG key): JAX random number generator state.
    ---------------------------------------------

    population (JAX array) (population_size, n_codons):
    [[0 4 1 2]
    [0 3 0 3]
    [0 4 0 3]
    [0 5 0 3]]

    where for sequence 1: [0 4 1 2], posiiton 0 chose synonymous codon option 0,
    posiiton 1 chose synonymous codon option 4 and so on for each sequence...
    -----------------------------------------------

    codon_choices_nt example (JAX array) (n_codons, max_sysnonymous_choices, 3):

    position 0: 3 codon choaices [0,0,0] and [1,1,1] etc..
    [[0, 0, 0],
     [1, 1, 1],
     [2, 2, 2]],

    position 1: 3 codon choices [0,1,0] and [1,2,1] adn so on 0,1,0 = ACA
    [[0, 1, 0],
     [1, 2, 1],
     [2, 3, 2]],

    so codon_choices_nt indicates the allowed codons at each position
    -----------------------------------------------
    syn_counts (JAX array) (n_codons,): number of codons, for the aa, available in that position
    -----------------------------------------------
    mutation_rate (float or JAX scalar float): chance of each codon position being randomply changed during mutation process.
    -----------------------------------------------
    elite_count (int): number of top performing sequences that survive selection.
    -----------------------------------------------
    bin_low (int): low end of optimized cpg count
    -----------------------------------------------
    bin_high (int): high end of optimized cpg count
    -----------------------------------------------
    target_metric: (str): cpg or gc

    """

    #cpg_counts = cpg_counts_from_population(population, codon_choices_nt)
    #cpg_counts_i32 = cpg_counts.astype(jnp.int32)

    target_counts = metric_counts_from_population(
        population,
        codon_choices_nt,
        target_metric)
    
    target_counts_i32 = target_counts.astype(jnp.int32)
    

    bin_low = jnp.asarray(bin_low, dtype=jnp.int32)
    bin_high = jnp.asarray(bin_high, dtype=jnp.int32)
    # Distance from target bin.
    # If CpG count is inside [bin_low, bin_high], distance = 0.
    # If below the bin, distance = bin_low - count.
    # If above the bin, distance = count - bin_high.

    distance_below = jnp.maximum(bin_low - target_counts_i32, 0)
    distance_above = jnp.maximum(target_counts_i32 - bin_high, 0)
    distance_to_bin = distance_below + distance_above

    # fitness in this algo is defined as highest when distance is small
    # inside bin sequences have fitness 0
    # outside bin sequences have negative fitness.

    # add small amount of random noise to break ties to avoid determanistic
    # selectio of sequences inside bin

    k_tiebreak, k_parent, k_mutation, k_new_choices = jax.random.split(key, 4)

    tie_noise = jax.random.uniform(
        k_tiebreak,
        shape=target_counts.shape,
        minval=0.0,
        maxval=1e-3
    )

    fitness = -distance_to_bin.astype(jnp.float32) + tie_noise
    top_values, top_indices = jax.lax.top_k(fitness, elite_count)
    elites = population[top_indices]
    elite_target_counts = target_counts_i32[top_indices]
    elite_distances = distance_to_bin[top_indices]

    best_target_count = elite_target_counts[0]
    best_distance = elite_distances[0]

    exact_hits = jnp.sum(distance_to_bin == 0)

    mean_target_count = jnp.mean(target_counts.astype(jnp.float32))
    mean_elite_target_count = jnp.mean(elite_target_counts.astype(jnp.float32))

    population_size = population.shape[0]
    n_codons = population.shape[1]

    parent_indices = jax.random.randint(
        k_parent,
        shape=(population_size,),
        minval=0,
        maxval=elite_count,
    )

    children = elites[parent_indices]

    mutation_mask = jax.random.bernoulli(
        k_mutation,
        p=mutation_rate,
        shape = (population_size, n_codons)
    )

    random_float = jax.random.uniform(
        k_new_choices,
        shape=(population_size, n_codons),
        minval=0.0,
        maxval=1.0
    )

    new_choices = jnp.floor(random_float * syn_counts[None, :]).astype(jnp.uint8)
    children = jnp.where(mutation_mask, new_choices, children)




    return children,best_target_count, best_distance, mean_target_count, mean_elite_target_count, exact_hits



@partial(jax.jit, static_argnames=("elite_count", "objective"))
def evolve_step(
    key,
    population,
    codon_choices_nt,
    syn_counts,
    mutation_rate,
    elite_count,
    objective,
):
    """

    same function as above but restricted to exploring the boundary space
    i.e. the min and max possible values for CpG.


    Simple EA:
    1. score population
    2. keep elite sequences
    3. sample parents from elites
    4. mutate synonymous codon choices
    """
    scores = cpg_counts_from_population(population, codon_choices_nt)

    if objective == "max":
        top_values, top_indices = jax.lax.top_k(scores, elite_count)
        best_score = top_values[0]
        mean_elite_score = jnp.mean(top_values)
    elif objective == "min":
        top_values, top_indices = jax.lax.top_k(-scores, elite_count)
        best_score = -top_values[0]
        mean_elite_score = jnp.mean(-top_values)
    else:
        raise ValueError("objective must be either 'max' or 'min'")

    elites = population[top_indices]

    k1, k2, k3 = jax.random.split(key, 3)

    population_size = population.shape[0]
    n_codons = population.shape[1]

    parent_indices = jax.random.randint(
        k1,
        shape=(population_size,),
        minval=0,
        maxval=elite_count,
    )

    children = elites[parent_indices]

    mutation_mask = jax.random.bernoulli(
        k2,
        p=mutation_rate,
        shape=(population_size, n_codons),
    )

    random_float = jax.random.uniform(
        k3,
        shape=(population_size, n_codons),
        minval=0.0,
        maxval=1.0,
    )

    new_choices = jnp.floor(random_float * syn_counts[None, :]).astype(jnp.uint8)

    children = jnp.where(mutation_mask, new_choices, children)

    mean_score = jnp.mean(scores)

    return children, best_score, mean_score, mean_elite_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cds", required=True, help="Path to KRAS CDS text file.")
    parser.add_argument("--population-size", type=int, default=100_000)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--elite-fraction", type=float, default=0.05)
    parser.add_argument("--mutation-rate", type=float, default=0.02)
    parser.add_argument("--bin-low", type=int, required=True)
    parser.add_argument("--bin-high", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-metric", choices=["cpg","gc"], required=True, help="Metric to optmize across bins: CpG count or GC count")
    parser.add_argument("--output-dir", type=str, default="outputs/metrics_batches", help="Directory to save per-run CpG/GC metric batches.")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    

    cds = read_sequence_cds(args.cds)
    codon_choices_nt_np, syn_counts_np, wt_choice_indices_np = build_synonymous_tables(cds)

    print(f"CDS length: {len(cds)} nt")
    print(f"Codons: {len(cds) // 3}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Target metric: {args.target_metric}")
    print(f"Target bin: {args.bin_low}-{args.bin_high}")
    print(f"Population size: {args.population_size}")
    print(f"Generations: {args.generations}")

    codon_choices_nt = jnp.asarray(codon_choices_nt_np, dtype=jnp.uint8)
    syn_counts = jnp.asarray(syn_counts_np, dtype=jnp.uint8)

    elite_count = max(1, int(args.population_size * args.elite_fraction))
    print(f"Elite count: {elite_count}")

    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)

    population = initialize_population(init_key, args.population_size, syn_counts)

    # Force compile - First run includes compilation.
    initial_cpg_counts = cpg_counts_from_population(population, codon_choices_nt)
    initial_gc_counts = gc_counts_from_population(population, codon_choices_nt)
    initial_cpg_counts.block_until_ready()
    initial_gc_counts.block_until_ready()


    print(
        "Initial CpG:",
        f"min={int(jnp.min(initial_cpg_counts))}",
        f"mean={float(jnp.mean(initial_cpg_counts)):.3f}",
        f"max={int(jnp.max(initial_cpg_counts))}",)

    print(
        "Initial GC:",
        f"min={int(jnp.min(initial_gc_counts))}",
        f"mean={float(jnp.mean(initial_gc_counts)):.3f}",
        f"max={int(jnp.max(initial_gc_counts))}")


    for generation in range(args.generations):
        key, step_key = jax.random.split(key)

        population, best_target_count, best_distance, mean_target_count, mean_elite_target_count, exact_hits = evolve_step_metric_bin(
            step_key,
            population,
            codon_choices_nt,
            syn_counts,
            args.mutation_rate,
            elite_count,
            args.bin_low,
            args.bin_high,
            args.target_metric)

        #gc_counts = gc_counts_from_population(population, codon_choices_nt)
        #gc_counts.block_until_ready()
        

        # Synchronize so printed timings/values are real
        best_target_count.block_until_ready()

        if generation % 5 == 0 or generation == args.generations - 1:
            cpg_counts = cpg_counts_from_population(population, codon_choices_nt)
            gc_counts = gc_counts_from_population(population, codon_choices_nt)

            cpg_counts.block_until_ready()
            gc_counts.block_until_ready()


            print(
                f"gen={generation:04d}",
                f"target={args.target_metric}",
                f"best_target={int(best_target_count)}",
                f"best_distance={int(best_distance)}",
                f"mean_target={float(mean_target_count):.3f}",
                f"elite_mean_target={float(mean_elite_target_count):.3f}",
                f"exact_hits={int(exact_hits)}",
                f"cpg_min={int(jnp.min(cpg_counts))}",
                f"cpg_mean={float(jnp.mean(cpg_counts)):.3f}",
                f"cpg_max={int(jnp.max(cpg_counts))}",
                f"gc_min={int(jnp.min(gc_counts))}",
                f"gc_mean={float(jnp.mean(gc_counts)):.3f}",
                f"gc_max={int(jnp.max(gc_counts))}")

    final_cpg_counts = cpg_counts_from_population(population, codon_choices_nt)
    final_gc_counts = gc_counts_from_population(population, codon_choices_nt)

    final_cpg_counts.block_until_ready()
    final_gc_counts.block_until_ready()


    metrics_path = output_dir / f"metrics_{args.target_metric}_{args.bin_low:03d}_{args.bin_high:03d}_seed{args.seed}.npz"
    np.savez_compressed(
        metrics_path,
        cpg_counts=np.asarray(final_cpg_counts),
        gc_counts=np.asarray(final_gc_counts),
        target_metric=args.target_metric,
        bin_low=args.bin_low,
        bin_high=args.bin_high,
        seed=args.seed,
        population_size=args.population_size,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        elite_fraction=args.elite_fraction,)

    print(f"Saved metric batch: {metrics_path}")




    print(
        "Final CpG:",
        f"min={int(jnp.min(final_cpg_counts))}",
        f"mean={float(jnp.mean(final_cpg_counts)):.3f}",
        f"max={int(jnp.max(final_cpg_counts))}")
    

    print(
    "Final GC:",
    f"min={int(jnp.min(final_gc_counts))}",
    f"mean={float(jnp.mean(final_gc_counts)):.3f}",
    f"max={int(jnp.max(final_gc_counts))}")


if __name__ == "__main__":
    main()




