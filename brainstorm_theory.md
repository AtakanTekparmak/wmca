# Theory + Literature Scan: Stabilizing Autoregressive Prediction on Chaotic Dynamics

Scope: methods that the WMCA project (frozen-reservoir + neural correction, "rescor") could plausibly borrow during a 1-week sprint to push deterministic rollout horizon past H~15 on chaotic continuous dynamics. Stochastic-latent (Dreamer-style) ideas are deliberately excluded.

---

## 1. The chaos-amplification framework: what it predicts

Mikhaeil, Monfared & Durstewitz (NeurIPS 2022, "On the difficulty of learning chaotic dynamics with RNNs") prove the obstacle is information-theoretic, not optimization-theoretic. If an RNN is trained on a chaotic orbit with maximal Lyapunov exponent lambda > 0, the loss-gradient norm through k steps grows as exp(k * lambda), and the *minimum-achievable* one-step error compounds at the same rate; the predictability horizon ~ (1/lambda) * log(1/eps) is a hard ceiling for any deterministic predictor (eps = irreducible state-uncertainty floor). They also show that constraining recurrence to be non-chaotic (orthogonal/Lipschitz <= 1) destroys the very dynamical regimes (multistability, chaos) the model is trying to reconstruct.

This makes three predictions for the WMCA exploration sprint:

- **Methods that only reduce gradient norm cannot lift the horizon ceiling.** Spectral normalization, orthogonal RNN, unitary RNN — they tame *training* gradients but cap expressivity below the target attractor; in the best case they buy stable but qualitatively-wrong rollouts.
- **Methods that reduce one-step error eps shift the horizon by log(1/eps) / lambda — sublinearly.** A 10x improvement in one-step MSE buys ~log(10)/lambda extra steps, not a 10x horizon. Bigger models / more data / better baselines therefore have diminishing returns.
- **Methods that reframe the loss away from per-step pointwise matching (shadowing, multistep penalty, invariant-statistics, refinement, denoising-during-rollout, diffusion-forcing, self-forcing) escape this ceiling on the *training-objective* axis** — they don't beat the Lyapunov clock pointwise, but they let you optimize trajectory-level / distributional / refinable quantities that *are* learnable past the horizon. These are the families to prioritize.

---

## 2. Survey by family

### 2.1 Diffusion forcing / Self forcing (training-time noise schedules per token)

- "Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion" — Chen, Sitzmann, Monsó, Du, Simchowitz, Tedrake (NeurIPS 2024).
- "Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion" — Huang, Li, He, Zhou, Shechtman (NeurIPS 2025 spotlight, arXiv 2506.08009).

**Idea.** Diffusion forcing trains a causal model to denoise each token at an *independently sampled noise level*, so future tokens can be queried at high noise (= less commitment) while past tokens are denoised heavily. Self-forcing then rolls out the model on its own outputs *during training*, with a video-level holistic loss and KV-caching, killing exposure bias directly.

**Relevance to chaos prediction: high (partial-strong).** Diffusion forcing's headline empirical claim is exactly what rescor needs: rollouts past the training horizon where deterministic baselines diverge. Self-forcing closes the loop; the user's deterministic-only constraint is satisfied if the noise levels are scheduled toward zero at inference (the diffusion is a training-time regularizer, not a stochastic latent). Note this is *related to but distinct from* PDE-Refiner (see 2.4) — diffusion forcing is per-token noise; PDE-Refiner is per-rollout-step refinement.

**Effort / compute.** Medium. Reference implementation exists (Boyuan Chen's repo). Per-token noise scheduling adds 1 axis to the dataloader; training cost is ~1.5-2x baseline. Self-forcing rollout-during-training adds K-step BPTT memory cost; controllable with stochastic gradient truncation.

**Prototype.** Add a per-token noise-level input to the rescor corrector head, train with mixed noise schedule + occasional self-rollout pushforward (see 2.7). Inference: zero noise.

### 2.2 Spectral normalization for recurrent stability

- "Spectral Normalization for Generative Adversarial Networks" — Miyato, Kataoka, Koyama, Yoshida (ICLR 2018).
- "Lipschitz Recurrent Neural Networks" — Erichson, Azencot, Queiruga, Hodgkinson, Mahoney (ICLR 2021).

**Idea.** Constrain the spectral norm (largest singular value) of recurrent weight matrices to <= 1 via power iteration; equivalently bound the local Jacobian's largest singular value, capping per-step gradient amplification.

**Relevance: partial / theoretically weak.** This is exactly the constraint Mikhaeil et al. warn against: enforcing Lipschitz-1 forbids chaotic attractors. For rescor specifically, the *frozen reservoir* is already (by design) above unit spectral radius — that's where chaos lives. Constraining the *trainable corrector* is fine and may help, but it cannot lift the horizon ceiling. Useful as a regularization-floor baseline, not a primary bet.

**Effort / compute.** Trivial. `torch.nn.utils.spectral_norm` works out of the box on linear/conv layers. Negligible compute overhead.

**Prototype.** Wrap rescor's corrector linear/conv layers with `spectral_norm`; treat as an ablation rather than the lead idea.

### 2.3 Lyapunov / Jacobian regularization

- "Jacobian Regularization Stabilizes Long-Term Integration of Neural Differential Equations" — Rojas et al., recent arXiv.
- "Stabilizing Equilibrium Models by Jacobian Regularization" — Bai, Koltun, Kolter (NeurIPS 2021).
- "JAWS: Enhancing Long-term Rollout of Neural PDE Solvers via Spatially-Adaptive Jacobian Regularization" — recent arXiv.

**Idea.** Add a loss term penalizing the Frobenius / spectral norm of the model Jacobian (or its directional derivative along trajectories), measured by Hutchinson trace estimators. Trains the model to be locally non-expanding *along realized trajectories* without enforcing a global Lipschitz constraint.

**Relevance: partial.** Subtler than 2.2 — by penalizing Jacobian norm only along the data manifold, you allow off-manifold contraction (which kills compounding error) while permitting on-manifold chaotic stretch. JAWS in particular targets exactly the autoregressive-PDE-rollout setting and reports horizon improvements on 2D systems. Compatible with the rescor freeze (apply only to corrector Jacobian).

**Effort / compute.** Low-medium. Hutchinson estimator adds ~1 extra backward per step (~2x training cost). Hyperparameter (regularization weight) needs tuning.

**Prototype.** Add `lambda_jac * ||J_corrector||_F^2` term, estimated stochastically, to the training loss; sweep lambda on the gs benchmark.

### 2.4 PDE-Refiner / iterative refinement during rollout

- "PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE Solvers" — Lippe, Veeling, Perdikaris, Turner, Brandstetter (NeurIPS 2023).
- "Message Passing Neural PDE Solvers" — Brandstetter, Worrall, Welling (ICLR 2022) (the *pushforward trick* + temporal bundling).

**Idea.** At each rollout step, run K denoising-style refinement passes with an exponentially-decaying noise schedule (min noise ~1e-7). Forces the model to attend to non-dominant spatial frequencies that vanilla MSE training drops; these are the very frequencies whose mis-modeling causes blow-up. Pushforward trick: at training time, occasionally feed the model its own (one-step) predictions as input — implicit noise injection that closes the train/test gap.

**Relevance: high.** PDE-Refiner is empirically the strongest "deterministic but with refinement loop" method in the chaotic-PDE literature; explicitly designed for long rollouts on Kuramoto–Sivashinsky / 2D Kolmogorov flow. Pushforward trick is dirt-cheap and a pure win in essentially every neural-PDE study since 2022.

**Effort / compute.** Pushforward trick: trivial (1 extra forward pass per training step, no architecture change). Full PDE-Refiner: medium (need a noise-conditional corrector + K refinement passes at inference, so ~Kx inference cost — K=4 typical).

**Prototype.** Phase 1: add pushforward trick to rescor (1 day). Phase 2: add noise-conditioned refinement head to corrector with exponential noise schedule.

### 2.5 Newer SSMs: S5, Mamba-2, GLA, RWKV

- "Resurrecting Recurrent Neural Networks for Long Sequences" — Orvieto et al. (LRU; ICML 2023).
- "Simplified State Space Layers for Sequence Modeling" — Smith, Warrington, Linderman (ICLR 2023, S5).
- "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality" — Dao, Gu (ICML 2024, Mamba-2).

**Idea.** Diagonal-plus-low-rank linear recurrences with HiPPO-derived initializations and (in Mamba/Mamba-2) input-dependent selectivity. By construction the recurrence is stable (eigenvalues of A inside or on the unit disk after discretization).

**Relevance: partial.** S5/Mamba-2 are *better encoders for long context*, not better predictors of chaos. Their built-in stability (linear, contractive) is the Mikhaeil-warning regime — they cannot themselves generate chaos, only encode it. Useful as a temporal-context stack *feeding* a chaotic reservoir or corrector. The user already tried per-cell Mamba; the upgrade path is Mamba-2 (SSD layer, faster, more stable batch-norm) or S5 (smaller, parallel scan, cleaner ablation).

**Effort / compute.** Low. Both have PyTorch reference implementations; S5 in particular is CPU-friendly via parallel scan in JAX/Triton (CPU fallback exists). Drop-in replacement for current Mamba blocks.

**Prototype.** Swap per-cell Mamba for S5 (smaller, faster on CPU) as a cheap ablation; if it's a wash, the issue is *not* the temporal context module — it's the rollout loss.

### 2.6 Koopman operators

- "Deep learning for universal linear embeddings of nonlinear dynamics" — Lusch, Kutz, Brunton (Nat. Commun. 2018).
- "Markov Neural Operators for Learning Chaotic Systems" / "Learning Dissipative Dynamics in Chaotic Systems" — Li, Liu-Schiaffini, Kovachki et al. (NeurIPS 2022).

**Idea.** Lift the state into a high-dim observable space where evolution becomes linear, then advance by matrix multiplication. For chaotic systems, the Koopman operator has continuous spectrum (no finite-dim invariant subspace) — practical methods learn an approximate finite-dim Koopman + dissipativity / Sobolev regularization to avoid blow-up off-attractor.

**Relevance: partial.** Koopman provides a principled framework (linear evolution in a learned latent) that is conceptually adjacent to "frozen reservoir + linear readout" — i.e., to the original Pathak et al. ESN setup. Markov Neural Operator's *dissipativity regularizer* (push the latent back toward attractor when it leaves a learned shell) is a clean drop-in safety net for any rollout. But the linear-Koopman assumption is provably insufficient for fully chaotic systems on a finite latent; expect rollout improvement from the dissipativity post-processing more than from Koopman itself.

**Effort / compute.** Medium. MNO has a public PyTorch repo. Dissipativity regularizer alone is ~30 lines.

**Prototype.** Add MNO-style dissipativity regularization (Sobolev loss + "push back into attractor shell" term) to rescor; the shell can be estimated as a per-channel quantile envelope from training trajectories.

### 2.7 Multistep penalty / shadowing / pushforward losses

- "Divide and Conquer: Learning Chaotic Dynamical Systems With Multistep Penalty Neural Ordinary Differential Equations" — Chakraborty et al. (CMAME 2024, arXiv 2407.00568).
- "Improved deep learning of chaotic dynamical systems with multistep penalty losses" — same group (arXiv 2410.05572, 2024).

**Idea.** Split the trajectory into non-overlapping windows; train each window independently (so BPTT depth never exceeds window length, taming gradient explosion); add a *discontinuity penalty* between adjacent windows so the resulting global trajectory remains smooth. Shown to be a cheap, scalable approximation of least-squares-shadowing (LSS) — provably the right loss for chaotic gradients.

**Relevance: high.** This is the most theoretically well-motivated chaos-specific training trick currently in the literature: it directly attacks Mikhaeil et al.'s exploding-BPTT problem without forcing non-chaos. Demonstrated on Kuramoto–Sivashinsky, 2D Kolmogorov flow, ERA5. Cleanly composable with any architecture (including frozen-reservoir + corrector).

**Effort / compute.** Low. Pure loss-function change + chunked BPTT. No architectural change. Slight VRAM reduction (shorter BPTT). Public code referenced.

**Prototype.** Reformulate rescor's training loss as: chunk trajectory into windows of length W (W ~ 1 / lambda_max ~ 5-10 steps for gs); train each chunk with teacher forcing + add `mu * ||x_window_end_predicted - x_next_window_start||^2` penalty.

### 2.8 Deep equilibrium / fixed-point methods (DEQ)

- "Deep Equilibrium Models" — Bai, Kolter, Koltun (NeurIPS 2019).
- "Stabilizing Equilibrium Models by Jacobian Regularization" — Bai et al. (NeurIPS 2021).

**Idea.** Replace a deep stack of layers with a single layer iterated to fixed point z* = f(z*, x); backprop via implicit differentiation (constant-memory). Pairs naturally with Jacobian regularization to ensure the fixed-point iteration converges.

**Relevance: low for autoregressive chaos.** DEQ is a *static* mapping (input -> equilibrium), not a temporal predictor; the rollout operator is f(x_t) = x_{t+1}, which is by assumption non-contractive on chaotic data — there is no fixed point to converge to. DEQ helps for tasks where the *answer* is a fixed point (e.g., flow estimation, solving an implicit operator), not where the answer is a chaotic trajectory. Possible niche use: replace a deep corrector stack with a DEQ corrector at *each* step (saving memory), but the chaos is in time, not depth. Skip for the sprint.

**Effort / compute.** Medium-high (root-finding, implicit diff). Not worth the cost given the mismatch.

### 2.9 Energy-based / Hopfield-style attractor priors

- "Hopfield Networks Is All You Need" — Ramsauer et al. (ICLR 2021).
- LeCun et al., "A Tutorial on Energy-Based Learning" (2006); recent EBM-AR equivalence work (arXiv 2512.15605).

**Idea.** Place an energy function over states / sequences; rollouts that drift away from the data manifold are penalized by rising energy and pulled back via gradient descent on E. Modern Hopfield networks give exponential storage capacity for attractor patterns and are equivalent to attention.

**Relevance: partial.** A learned energy E(x) on the *attractor* of the chaotic system is essentially the dissipativity regularizer of MNO (2.6) in disguise — it stabilizes off-manifold drift, the dominant rescor failure mode at H>15, without restricting on-manifold chaos. Attractive theoretically. The catch: training a stable energy that's flat *on* the attractor and steep *off* it is finicky on continuous-state chaotic systems; most demonstrations are on discrete or quasi-periodic data.

**Effort / compute.** Medium. Auxiliary energy head + Langevin-style correction step at inference. Risk of new instabilities (Langevin step size).

**Prototype.** Train an autoencoder reconstruction loss as a soft energy proxy, project rollout states back to the manifold every K steps. Cheaper than full EBM training and captures the same intuition.

### 2.10 Deep ensembles for chaotic prediction

- "Data-driven forecasting of high-dimensional chaotic systems with long short-term memory networks" — Vlachas et al. (Proc. Royal Soc. A 2018).
- "GenCast" / "Probabilistic weather forecasting with machine learning" — Price et al. (Nature 2024).

**Idea.** Train K independent rescor models with different seeds/initializations; rollout the ensemble; report ensemble mean (or member trajectory closest to ensemble mean — "central member") as the deterministic prediction. Ensemble disagreement quantifies the Lyapunov-derived predictability horizon directly.

**Relevance: partial-good.** Ensembles cannot beat the per-member horizon; what they buy is (a) a *honest* per-step uncertainty estimate that lets you stop rollout when the ensemble explodes, (b) a smoothed central trajectory that often outlives any individual member, and (c) reduced compounding noise floor (eps shrinks as 1/sqrt(K)). GenCast's central insight is exactly this for weather. For rescor's deterministic constraint, the *mean* of the ensemble is itself a deterministic prediction.

**Effort / compute.** Trivial conceptually but K-fold compute. K=5 ensemble probably gives ~15-25% horizon extension at 5x training+inference cost.

**Prototype.** Train 5 rescor seeds; report mean trajectory. Cheap sanity check that rules out / rules in the "noise floor was the bottleneck" hypothesis.

### 2.11 Consistency models (for completeness — covered briefly)

- "Consistency Models" — Song, Dhariwal, Chen, Sutskever (ICML 2023).

**Idea.** Train a model that maps any point along the diffusion ODE trajectory directly to the data endpoint, enabling 1-step generation.

**Relevance: low for the autoregressive setting per se, partial as an accelerator for diffusion-forcing (2.1).** Consistency models compress an N-step diffusion sampler into 1-2 steps; they'd matter if diffusion forcing's per-step refinement cost were the bottleneck, which on CPU it might be. Treat as a stage-2 follow-on to 2.1, not a primary lead.

**Effort / compute.** Medium-high; consistency training is finicky.

---

## 3. Top-3 picks for the 1-week sprint

Ranked by expected horizon-improvement-per-engineering-week, conditional on "the user wants deterministic rollout to survive H=15 -> H=100 on gs / Crafter latents".

### Pick #1: Multistep penalty loss (Section 2.7)

- **Why it transfers.** Direct theoretical attack on the Mikhaeil et al. failure mode: it bounds BPTT depth (so gradients stay finite) without bounding the Lyapunov exponent of the recurrence (so chaos remains expressible). Already validated on KS / 2D Kolmogorov / ERA5 — closer to gs/Crafter dynamics than any of the language/vision-domain alternatives.
- **Why first.** Pure loss function change. Zero architectural risk. ~1 day to implement, 2-3 days to sweep window-length W and penalty mu.
- **Risk.** Small. Worst case: matches teacher-forcing baseline.

### Pick #2: PDE-Refiner pushforward trick + denoising-during-rollout (Section 2.4)

- **Why it transfers.** PDE-Refiner specifically diagnoses *high-frequency under-modeling* as the long-rollout failure mode in chaotic PDE solvers. gs/Crafter latents are exactly that domain (continuous, multi-scale, deterministic). The pushforward trick alone fixes exposure bias for free.
- **Why second.** Requires architectural change (noise-conditional corrector head). 2-3 days for refinement, 1 day for pushforward.
- **Risk.** Medium. Depends on whether high-frequency mis-modeling is actually the rescor failure mode (worth a 1-day diagnostic of FFT spectra of predicted vs ground-truth at H=10,15,20).

### Pick #3: Diffusion forcing (Section 2.1)

- **Why it transfers.** Empirically the strongest published method for past-training-horizon deterministic rollouts in continuous video. Compatible with rescor's frozen-reservoir backbone (per-token noise level is a corrector input, not a reservoir change). Subsumes pushforward + PDE-Refiner as special cases of a noise schedule.
- **Why third (not first).** More moving parts, more hyperparameters, larger code delta than #1 and #2. If #1 alone gets to H=50, this may be unnecessary; if #1+#2 plateau at H=30, this is the natural escalation.
- **Risk.** Medium-high engineering, low theoretical.

**Suggested sprint order.** Day 1-2: implement #1 (multistep penalty) end-to-end, run gs ablation. Day 3: pushforward-only variant of #2. Day 4-5: full PDE-Refiner-style refinement. Day 6-7: if needed, full diffusion forcing.

---

## 4. Negative findings — methods that look promising but theoretically can't help much

### Bigger models / more data
Mikhaeil et al. imply log(1/eps) / lambda dependence — improving eps by a factor of 10 yields ~2.3 / lambda extra horizon steps, sublinearly. The user's 16 baselines and current model size already saturate this term for gs.

### Spectral normalization on the recurrence (rescor reservoir or corrector recurrence)
Provably forbids chaotic attractor reconstruction. Useful only on *non-recurrent* corrector sublayers as a regularizer floor; do not apply to anything on the temporal axis.

### Orthogonal / unitary RNN parameterizations
Same failure mode: enforces Lipschitz <= 1 globally. Will give numerically stable but qualitatively wrong (non-chaotic) rollouts.

### Bigger context / longer Mamba windows
For *Markov* chaotic dynamics (which gs is, by construction), no finite history beyond the Markov order improves the optimal one-step predictor. Crafter latents likely have small effective Markov order; expect minor improvement from doubling Mamba window, not a horizon-class change.

### Plain teacher forcing with longer BPTT
Mikhaeil et al. is essentially a no-go theorem for this regime: longer BPTT *exponentially* worsens the gradient pathology rather than helping. Sparse-forcing BPTT (their proposal) helps; plain longer BPTT does not.

### DEQ / deep equilibrium for the rollout operator itself
Chaos has no fixed point in the time map by definition. DEQ is the wrong abstraction for this problem.

### Pure consistency-model distillation
Compresses a sampler, doesn't change the underlying training signal. Only helpful as a post-hoc accelerator if diffusion forcing (Pick #3) itself works.

### Plain ensemble averaging without re-training
Ensembling K seeds of a *failing* deterministic predictor smooths the central trajectory but cannot lift the fundamental horizon ceiling. Useful diagnostic, not a standalone fix.

---

## References (key papers, by section)

- Mikhaeil, Monfared, Durstewitz. *On the difficulty of learning chaotic dynamics with RNNs*. NeurIPS 2022.
- Chen, Sitzmann, Monsó, Du, Simchowitz, Tedrake. *Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion*. NeurIPS 2024.
- Huang, Li, He, Zhou, Shechtman. *Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion*. NeurIPS 2025.
- Lippe, Veeling, Perdikaris, Turner, Brandstetter. *PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE Solvers*. NeurIPS 2023.
- Brandstetter, Worrall, Welling. *Message Passing Neural PDE Solvers*. ICLR 2022 (pushforward trick).
- Chakraborty et al. *Divide and Conquer: Learning Chaotic Dynamical Systems With Multistep Penalty Neural ODEs*. CMAME 2024 (arXiv 2407.00568) and follow-up arXiv 2410.05572.
- Li, Liu-Schiaffini, Kovachki et al. *Learning Dissipative Dynamics in Chaotic Systems* (Markov Neural Operator). NeurIPS 2022.
- Erichson, Azencot, Queiruga, Hodgkinson, Mahoney. *Lipschitz Recurrent Neural Networks*. ICLR 2021.
- Miyato, Kataoka, Koyama, Yoshida. *Spectral Normalization for GANs*. ICLR 2018.
- Bai, Kolter, Koltun. *Deep Equilibrium Models*. NeurIPS 2019.
- Bai, Koltun, Kolter. *Stabilizing Equilibrium Models by Jacobian Regularization*. NeurIPS 2021.
- Ramsauer et al. *Hopfield Networks Is All You Need*. ICLR 2021.
- Song, Dhariwal, Chen, Sutskever. *Consistency Models*. ICML 2023.
- Smith, Warrington, Linderman. *Simplified State Space Layers for Sequence Modeling* (S5). ICLR 2023.
- Dao, Gu. *Transformers are SSMs (Mamba-2)*. ICML 2024.
- Pathak, Hunt, Girvan, Lu, Ott. *Model-Free Prediction of Large Spatiotemporally Chaotic Systems from Data*. PRL 2018.
- Price et al. *Probabilistic Weather Forecasting with Machine Learning* (GenCast). Nature 2024.
