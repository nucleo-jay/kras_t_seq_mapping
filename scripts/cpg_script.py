import argparse
import os
from functools import partial
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
import jax.numpy as jnp
import numpy as np
from data.tables import codon_to_aa

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


def build_synonymous_tables(cds: str):#
    """Returns:
    codon_choices_nt """

    codons = [cds[i:i+3] for i in range(0, len(cds), 3)]
    aa_seq = [codon_to_aa[i] for i in codons]

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

    seq = population_to_nt(population, codon_choices_nt) # get sequence from above function
    cytosine_mask = seq[:, :-1] == 1 # mask for places where 1 (C)
    guanine_mask = seq[:, 1:] == 2 # mask for palces where 2 (G)

    cpg_sites = cytosine_mask & guanine_mask # look for places with combined mask C --> G


    return jnp.sum(cpg_sites, axis=1)



@partial(jax.jit, static_argnames=("population_size",))
def initialize_population(key, population_size, syn_counts):
    """ 
    key (int): JAX random number generator state
    population_size (int): number of sequences in the population
    syn_counts: 1D array defining the number of synonymous choices at each position of the sequence - derived from build_synonymous_tables()
    e.g. syn_counts = [1, 6, 2, 4]: position 1 has 1 synonymous choice, 
    position 2 has 6 synonymous choices,
    posiiton 3 has 2 synonymous choices and so on...

    """

    n_codons = syn_counts.shape[0]
    random_float = jax.random_uniform(
        key,
        shape=(population_size, n_codons),
        minval=0.0,
        maxval=1.0
    )

    population = jnp.floor(random_float * syn_counts[None, :]).astype(jnp.uint8)

    return population









test_cds = read_sequence_cds("./data/principal_sequence.txt")

#out = build_synonymous_tables(test_cds)
#print(out)


codon_choices_nt = jnp.array([
    [[1, 2, 1],
     [1, 1, 2],
     [2, 2, 2]],

    [[0, 1, 2],
     [1, 2, 1],
     [1, 2, 2]],
], dtype=jnp.uint8)

population = jnp.array([
    [0, 0],
    [1, 2],
    [2, 1],
], dtype=jnp.int32)

out = cpg_counts_from_population(population, codon_choices_nt)

print("type(out):", type(out))
print("out:")
print(out)

