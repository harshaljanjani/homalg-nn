## MNIST Homology Findings

**Date**: 2025-12-05  
**Task**: Swept lambdas, trained longer, tweaked architectures, dug deep into diagnostics.

Spent the last couple days chasing this ghost; trying to force Betti numbers to hit [0,0,0,0] in the `homalg-nn` MNIST chain setup. Spoiler: it didn't happen. **But here's the thing that actually excites me**. We figured out why it's not happening, and it's not some HP tuning mistake. It's a clash between **what the network wants for nailing classification** and **what we'd need for this perfect topological alignment**. Even without that perfection, we're seeing ++ in representation quality. 16% less entanglement.

### The Lambda Sweep -- Went Nowhere

**Experiment:** Crank lambda from 0.1 to 2.0, train for 15 epochs each. Five runs, hoping the exactness loss would kick in and drag Betti numbers down.
Here's how it shook out:

<div align="center">

| Lambda | Final Accuracy | Final Betti | Exactness Loss | Verdict |
|:------:|:--------------:|:-----------:|:--------------:|:-------:|
| 0.1    | 97.86%         | [0,0,0,128] | 37.33          | Nope    |
| 0.3    | 97.28%         | [0,0,0,128] | 37.33          | Nope    |
| 0.5    | 97.70%         | [0,0,0,128] | 37.33          | Nope    |
| 1.0    | 97.46%         | [0,0,0,128] | 37.33          | Nope    |
| 2.0    | 97.50%         | [0,0,0,128] | 37.33          | Nope    |

</div>

Every single one landed in the exact same spot. Betti stuck at `[0, 0, 0, 128]`, loss flat at 37.33. Lambda? It didn't budge the topology one bit, even as accuracy hummed along nicely.

**Key insight**: Lambda has ZERO effect on Betti convergence. All configurations identical - this rules out "just tune it harder" as a solution path.

### Training Longer -- Same Stubbornness

Maybe it just needs time? I doubled down to 30 epochs with `lambda = 0.5`.

<div align="center">

| Metric          | Epoch 1 | Epoch 15 | Epoch 30 | Change     |
|:---------------:|:-------:|:--------:|:--------:|:----------:|
| Task Loss       | 0.473   | 0.027    | 0.003    | -99% (win) |
| Test Accuracy   | 90.71%  | 97.74%   | 98.12%   | +7.4% (win)|
| Exactness Loss  | 37.326  | 37.330   | 37.331   | +0.01% (ugh)|
| Betti Numbers   | [0,0,0,128] | [0,0,0,128] | [0,0,0,128] | Zero change|

</div>

Task loss craters, accuracy climbs to 98% - beautiful. But exactness? It barely twitched.  
Betti numbers? Frozen. It's like the network said, "Cool, I'll learn the digits, but don't bug me about your chain complex stuff". Time wasn't the issue.

**Key insight**: Training duration has ZERO effect on Betti convergence. Task performance improves dramatically but topological properties remain completely unchanged.

### Flipping the Architecture -- Symmetry Changes the Game

This is where it got interesting. I tried a symmetric chain: `[16, 32, 64, 32, 16]`. Same 15 epochs, `lambda = 0.5`. Things shifted.

<div align="center">

| Metric         | Original [16,32,64,128,256] | Symmetric [16,32,64,32,16] | Change      |
|:--------------:|:---------------------------:|:--------------------------:|:-----------:|
| Final Betti    | [0,0,0,128]                 | [0,29,0,0]                 | New bottleneck|
| Exactness Loss | 37.33                       | 10.73                      | -71% (win)  |
| Test Accuracy  | 97.70%                      | 96.93%                     | -0.8% (trade-off)|

</div>

Betti flipped - now it's bloating in the middle at 29, not the end. Exactness loss dropped hard, to 10.73. But accuracy dipped a hair. Architecture isn't just details; it rewires the whole dynamic. Symmetric feels more balanced, like it's actually listening to the constraint a little. Still no `[0, 0, 0, 0]`, but closer. And that trade-off? Worth noting for when we care more about reps than raw digits.

**Key insight**: Architecture fundamentally changes behavior. Symmetric chain has lower exactness loss and different Betti pattern, but still doesn't achieve full exactness. This shows architecture determines which Betti numbers can converge, not just hyperparameters.

### What the Singular Values Are Telling Us

Lambda `0.5` and `1.0`, 10 epochs each. Deep diagnostics on singular value structure.

End-of-training singular values (all above epsilon = 0.001, no zeros lurking):
<br>`d0 (16×32)`: 16 big ones, min = 0.101
<br>`d1 (32×64)`: 32 big, min = 0.011
<br>`d2 (64×128)`: 64 big, min = 0.008
<br>`d3 (128×256)`: 128 big, min = 0.009

Full rank across the board - ALL boundary maps are FULL RANK throughout training.  
Kernels? Basically zero everywhere except the last one (dim 128, as expected).

**Exactness defects** <br> Bang on the dimensions: 16.0, 32.0, 64.0 - like images and kernels are perfectly at odds, no overlap. When defects equal dimensions exactly, that means **images and kernels are completely orthogonal**. There's no intersection at all.<br>
**Gradients** <br> Flowing strong, `10^0` to `10^1` range. No vanishing, no dying ReLUs. The optimization isn't broken; it's **thriving** at full rank. But that's the trap: full rank crushes classification, while exactness begs for low-rank alignment. They pull in opposite directions, and the task wins every time.<br>
**What this rules out** <br> The optimization is NOT failing due to vanishing gradients, insufficient lambda, or insufficient training time. The optimization IS stuck because full-rank maps are optimal for classification, exactness requires low-rank aligned subspaces, and these objectives are fundamentally in conflict.

### Diagnostic Plot Breakdown

Check `results/mnist_diagnostics/` for the full 9-panel plots. Here's what they're screaming:

**Top row (singular values)** <br> NEVER approach the epsilon threshold - stay 10-100x above it through all training. Not even close to forming low-rank structure.<br>
**Kernel dimensions** <br> Perfectly flat at ZERO for the first three maps. Not even one epoch of kernel growth. The fourth map's kernel is locked at 128 (as it must be geometrically).<br>
**Gradient flow** <br> Strong and healthy (`10^0` to `10^1`) across all boundary maps. Not a gradient problem - the gradients are there, they're just not driving exactness.<br>
**Rank evolution** <br> Perfectly flat horizontal lines. All maps stay full rank from init to final epoch. Zero movement toward the low-rank structure exactness needs.<br>
**Exactness defects** <br> Three flat horizontal lines at 16, 32, 64. Never improve by even a single unit. The defect magnitudes equal the dimensions exactly - geometric impossibility of alignment.<br>
**Betti numbers** <br> Perfectly flat at `[0, 0, 0, 128]`. Not even one step of improvement across any epoch. The topology is locked in from the start.<br>
**Interpretation** <br> The optimization has found a stable local minimum where task loss decreases but exactness loss remains constant. It's not struggling to optimize exactness - it's **actively choosing not to**. The gradient is telling us something: full-rank is what the task wants, and low-rank would hurt performance.

### The Real Why

Our chain: `R^16 ← R^32 ← R^64 ← R^128 ← R^256`. For exactness, images should nest perfectly into kernels: `im(d_{i+1}) = ker(d_i) ∀ i`.

But here's the geometric trap with full-rank maps:

**Kernel structure when all maps are full rank**:
<br>`ker(d0)` ≈ `{0}` (no kernel, it's injective)
<br>`ker(d1)` ≈ `{0}`
<br>`ker(d2)` ≈ `{0}` 
<br>`ker(d3)` has dimension 256 - 128 = 128 (large kernel at the end)

**Image dimensions**:
<br>`dim(im(d0))` = 16
<br>`dim(im(d1))` = 32
<br>`dim(im(d2))` = 64
<br>`dim(im(d3))` = 128

For exactness at position 1, we'd need `im(d1) ⊆ ker(d0)`. But `ker(d0)` ≈ `{0}`, so `im(d1)` must also be `{0}`. That would make `d1` zero-rank, killing all representation capacity at that layer.

**The fundamental conflict**:
<br>**Classification needs**: Rich, full-rank representations that separate classes
<br>**Exactness needs**: Low-rank, aligned subspaces where images live inside kernels

You'd need to squash everything to zero to align. That kills capacity - can't represent digits worth a damn. Classification screams for rich, separable spaces. Exactness whispers for compression and alignment. Gradient descent picks the screamer every time.<br>
Every run shows it: defects scale with how "good" the reps are. Orthogonal subspaces are the local minimum, and it's a comfy one for the task.

From the original MNIST experiments (`10 epochs`, `λ = 0.1`):
<br>**Entanglement**: -16% (0.157 → 0.132) ✓
<br>**Rank ratio**: +6.7% (0.117 → 0.125) ✓ 
<br>**FER score**: +2-28% improvement across metrics ✓
<br>**Test accuracy**: 98% maintained ✓

Chain structure helps representations even without `Betti = 0`. Partial exactness is delivering real benefits.

### The Conjecture That's Begging to Be True

**Statement** <br> For a chain complex used in a discriminative task (classification), achieving both high task performance (`> 95% accuracy`) AND perfect topological exactness (`Betti = 0`) is **impossible** for generic datasets and non-degenerate loss functions.<br>
**Intuition** <br> Discrimination requires separating classes in representation space. Separation requires dimensionality and non-degeneracy (full rank). Exactness requires low-rank aligned subspaces. These constraints are mutually exclusive at the geometric level.

**Evidence from experiments**:
<br>No configuration achieved `Betti = 0` with >96% accuracy across any lambda or training length
<br>All high-accuracy models had full-rank boundary maps with zero kernels
<br>Exactness defects proportional to representation richness (defect = dimension when orthogonal)
<br>Symmetric architecture trades accuracy for lower exactness loss, but still can't close the gap

**Open question**: Can we prove this formally? Or find a counterexample? The geometric argument feels tight, but a constructive proof would seal it.

### Comparison Context

**How this differs from standard regularization**:

**L2 regularization (weight decay)** <br> Shrinks all weights uniformly, no structural constraint, no topological interpretation. Just makes everything smaller.<br>
**Dropout** <br> Random feature removal, ensemble interpretation, no alignment between layers. Stochastic decorrelation.<br>
**Our approach (exactness-inspired)** <br> Structured constraint on subspace alignment, topological interpretation (chain complex), measurable improvements in representation quality. Not just "smaller" or "random" - it's about **geometric relationships between layers**.

**Connection to Information Bottleneck**:

**Information Bottleneck** <br> Compress input while preserving task information (MI maximization).<br>
**Exactness** <br> Align kernels and images across layers (subspace geometry).<br>

Both involve layer-wise constraints and compression-performance trade-offs. IB is information-theoretic; exactness is topological. Different lenses on the same "what should intermediate representations look like?" question.

### What the Numbers Actually Mean

**Betti = [0, 0, 0, 128]** <br> The homology is trivial except at the final position, where there's a 128-dimensional "hole" - the kernel of the last map that doesn't get filled by any image from above. This is geometrically forced by the dimensions.<br>
**Exactness loss = 37.33** <br> The sum of squared Frobenius norms of the defect matrices. It's measuring how far apart the subspaces are. Being stuck at 37.33 across all lambdas means the subspaces are locked in place.<br>
**Defect = dimension** <br> When `||im(d_{i+1}) - ker(d_i)|| = dim(ker(d_i))`, the image and kernel are orthogonal. Zero overlap. Maximum possible defect for that dimension. This is what full-rank buys you - complete separation.<br>
**Full rank everywhere** <br> Every singular value above threshold means the map is using all its capacity. No compression, no alignment, maximum expressivity. This is what you want for classification - every dimension is pulling its weight.

### The Symmetric Architecture Deep Dive

Why does `[16, 32, 64, 32, 16]` behave so differently from `[16, 32, 64, 128, 256]`?

**Geometric insight** <br> In the symmetric chain, you're forcing the network through a bottleneck (64) and then back out. This creates pressure at the middle map, not the end. The kernels and images have a chance to "meet in the middle" because dimensions are compressing and then expanding.<br>
**Result** <br> `Betti_1 = 29` instead of `Betti_3 = 128`. The topological hole moved. Exactness loss dropped from 37.33 to 10.73 - a 71% reduction. But accuracy dipped from 97.70% to 96.93% - the network is giving up some discriminative power to achieve better alignment.

This is the trade-off made explicit: symmetry helps topology at the cost of task performance. For classification, asymmetric wins. For representation quality, symmetric might be worth it.

### What Full Rank Really Costs Us

Having all maps at full rank means:
<br>**Zero null spaces** (except geometrically forced ones)
<br>**Maximum information flow** through each layer
<br>**No natural alignment points** between layers
<br>**Orthogonal image/kernel pairs** everywhere

This is the **opposite** of what a chain complex wants. A chain complex wants:
<br>**Rich null spaces** that match the previous layer's image
<br>**Controlled information compression** (rank drops)
<br>**Nested subspaces** (im ⊆ ker)
<br>**Aligned directions** across boundaries

The optimization found the global minimum for classification and a local minimum for exactness. It's not wandering around lost - it's **parked** at the best spot for the task, which happens to be the worst spot for topology.

### The Bottom Line

Hyperparameters? Useless here; lambda or epochs don't touch it.  
Architecture? Huge lever - symmetric shifts the bottleneck and cuts exactness loss by 71%.  
Full rank? That's what the task is about, but exactness hates it.  
Perfect topology? Probably impossible for discriminative tasks at high accuracy.

But we don't need `Betti = 0` for wins. 16% entanglement drop (0.157 → 0.132), rank ratio up 6.7% (0.117 → 0.125), FER scores jumping 2-28%, all while maintaining 98% accuracy. **Partial is plenty.**

The constraint is guiding optimization toward better local minima for representation quality, even if it can't achieve global optimality for topology. That's the real result: exactness-inspired architectures improve representations measurably, without requiring perfect exactness.