import numpy as np
from homalg_nn.core import ChainComplex

print("Chain Complex Tutorial")

# =================================
# Trivial Chain Complex
# =================================
print("\n1. TRIVIAL CHAIN COMPLEX")
print("Creating a chain complex with all zero maps: R^2 <- R^3 <- R^2")

chain_trivial = ChainComplex(dimensions=[2, 3, 2])
print(f"\nChain structure: {chain_trivial}")
print(f"Valid chain complex: {chain_trivial.validate_chain_axiom()}")
print(f"Betti numbers: {chain_trivial.get_betti_numbers()}")
print(f"Is exact: {chain_trivial.is_exact()}")
print("\nInterpretation:")
print("  - Zero boundary maps trivially satisfy d o d = 0")
print("  - Betti numbers are [3, 2]: dimensions of the kernel of each zero map")
print("  - NOT exact because ker(d_i) != im(d_{i+1}) (one is the full space, other is {0})")

# =================================
# Short Exact Sequence
# =================================
print("\n\n2. SHORT EXACT SEQUENCE")
print("Creating the exact sequence: 0 -> R^2 --(i)--> R^3 --(p)--> R^1 -> 0")
print("  where i(x, y) = (x, y, 0) [inclusion]")
print("  and p(x, y, z) = z [projection]")

# inclusion map: R^2 -> R^3
i = np.array([
    [1, 0],
    [0, 1],
    [0, 0]
])
# projection map: R^3 -> R^1
p = np.array([[0, 0, 1]])
# chain: R^1 <- R^3 <- R^2
chain_exact = ChainComplex(dimensions=[1, 3, 2], boundary_maps=[p, i])
print(f"\nChain structure: {chain_exact}")
print(f"\nBoundary map d_0 (projection):")
print(p)
print(f"\nBoundary map d_1 (inclusion):")
print(i)
composition = p @ i  # should be zero
print(f"\nComposition d_0 o d_1 = p o i:")
print(composition)
print(f"Is zero: {np.allclose(composition, 0)}")
print(f"\nValid chain complex: {chain_exact.validate_chain_axiom()}")
print(f"Exact at R^3: {chain_exact.is_exact_at(degree=0)}")
print(f"Is exact everywhere: {chain_exact.is_exact()}")
print(f"Betti numbers: {chain_exact.get_betti_numbers()}")
print("\nDetailed analysis:")
beta_0, generators_0 = chain_exact.compute_homology(degree=0)
print(f"  H_0 = ker(p) / im(i):")
print(f"    dim(ker(p)) = {chain_exact.get_nullities()[0]}")
print(f"    dim(im(i)) = {chain_exact.get_ranks()[1]}")
print(f"    Betti number beta_0 = {beta_0}")
print("\nInterpretation:")
print("  - ker(p) = {(x, y, 0) : x, y in R} (2-dimensional)")
print("  - im(i) = {(x, y, 0) : x, y in R} (2-dimensional)")
print("  - ker(p) = im(i), so the sequence is exact!")
print("  - beta_0 = 0 confirms exactness: no 'holes' in the chain")

# =================================
# Non-Exact Sequence
# =================================
print("\n\n3. NON-EXACT SEQUENCE")
print("Creating a sequence with 'holes': R^2 <- R^3 <- R^2")
print("  d_0: (x, y, z) |-> (x, y) [project to first two coordinates]")
print("  d_1: zero map")

d_0 = np.array([[1, 0, 0], [0, 1, 0]])
d_1 = np.zeros((3, 2))
chain_nonexact = ChainComplex(dimensions=[2, 3, 2], boundary_maps=[d_0, d_1])

print(f"\nChain structure: {chain_nonexact}")
print(f"\nd_0 (projection):")
print(d_0)
print(f"\nd_1 (zero map):")
print(d_1)
print(f"\nValid chain complex: {chain_nonexact.validate_chain_axiom()}")
print(f"Exact at R^3: {chain_nonexact.is_exact_at(degree=0)}")
print(f"Betti numbers: {chain_nonexact.get_betti_numbers()}")
# exactness defect
defect_subspace = chain_nonexact.get_exactness_defect(0, method='subspace_distance')
defect_dim = chain_nonexact.get_exactness_defect(0, method='dimension_gap')
defect_proj = chain_nonexact.get_exactness_defect(0, method='projection_norm')
print(f"\nExactness defects (larger = further from exact):")
print(f"  Subspace distance: {defect_subspace:.6f}")
print(f"  Dimension gap: {defect_dim}")
print(f"  Projection norm: {defect_proj:.6f}")
beta_0, generators_0 = chain_nonexact.compute_homology(degree=0)
print(f"\nHomology analysis:")
print(f"  H_0 = ker(d_0) / im(d_1):")
print(f"    dim(ker(d_0)) = {chain_nonexact.get_nullities()[0]}")
print(f"    dim(im(d_1)) = {chain_nonexact.get_ranks()[1]}")
print(f"    Betti number beta_0 = {beta_0}")
print(f"  Homology generators (basis for H_0):")
print(generators_0)
print("\nInterpretation:")
print("  - ker(d_0) = {(0, 0, z) : z in R} (1-dimensional)")
print("  - im(d_1) = {0} (0-dimensional)")
print("  - ker(d_0) != im(d_1), so NOT exact")
print("  - beta_0 = 1: there's a 1-dimensional 'hole' in the chain")
print("  - The homology generator shows the direction of the hole")

# =================================
# Random Chain Complex
# =================================
print("\n\n4. RANDOM CHAIN COMPLEX")
print("Creating a random chain complex and analyzing its homology")
np.random.seed(42)

# create boundary maps that satisfy chain axiom
# `d_1` s.t. `im(d_1)` is a subset of `ker(d_0)`
d_0_random = np.random.randn(3, 5)
from homalg_nn.core.homology import compute_kernel
ker_dim, ker_basis = compute_kernel(d_0_random)
print(f"\nCreated random d_0: R^5 -> R^3")
print(f"  dim(ker(d_0)) = {ker_dim}")

# create d_1 that maps into ker(d_0)
# d_1: R^4 -> R^5, and we want im(d_1) subset ker(d_0)
# We can do this by: `d_1 = ker_basis @ A`, where A is (ker_dim × 4)
if ker_dim > 0:
    A = np.random.randn(ker_dim, 4)
    d_1_random = ker_basis @ A
else:
    # if ker is trivial, `d_1` must be zero
    d_1_random = np.zeros((5, 4))
chain_random = ChainComplex(dimensions=[3, 5, 4], boundary_maps=[d_0_random, d_1_random])

print(f"\nChain: R^3 <- R^5 <- R^4")
print(f"Valid chain complex: {chain_random.validate_chain_axiom()}")
summary = chain_random.summary()
print(f"\nSummary:")
print(f"  Dimensions: {summary['dimensions']}")
print(f"  Betti numbers: {summary['betti_numbers']}")
print(f"  Ranks: {summary['ranks']}")
print(f"  Nullities: {summary['nullities']}")
print(f"  Exact: {summary['is_exact']}")
print(f"  Exactness defects: {[f'{d:.6f}' for d in summary['exactness_defects']]}")
print("\nInterpretation:")
if summary['is_exact']:
    print("  - The chain is exact! ker = im at all degrees")
    print("  - All Betti numbers are zero (no holes)")
    print("  - Information flows through the chain without loss")
else:
    print("  - The chain is NOT exact")
    print(f"  - Betti numbers {summary['betti_numbers']} indicate homological defects")
    print("  - Non-zero homology reveals 'fractured' information flow")

# =================================
# Connection to FERs
# =================================
"""
5. CONNECTION TO FERs
How homology relates to neural network representations:
In a neural network modeled as a chain complex:
- Layers are vector spaces `C_i` (feature spaces)
- Weight matrices are boundary maps `d_i`
- Exactness means clean information flow with no redundancy
FER (Fractured Entangled Representation) occurs when:
- `beta_i > 0`: Homology detects redundant / fractured representations
- `ker(d_i)` properly contains `im(d_{i+1})`: Layer i has "extra capacity" not explained.
- This extra capacity allows the network to learn the same concept
  in multiple disconnected ways (fracturing).
Enforcing exactness (`beta_i = 0`) "hopefully" prevents FERs by:
1. Forcing unified representations (no redundancy)
2. Ensuring factored structure (orthogonal capabilities stay orthogonal)
3. Enabling compositional generalization
"""
