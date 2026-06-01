<p align="center">
  <img src="assets/teaser.png" alt="MORPHOS teaser: given video inputs, MORPHOS autoregressively generates unified dynamic 3D representations — meshes, 3D Gaussians, and radiance fields." width="100%">
</p>

# MORPHOS: Autoregressive 4D Generation with Temporal Structured Latents

<p align="center">
  <a href="https://mkxdxdxd.github.io/">Minkyung Kwon</a><sup>*</sup> ·
  <a href="https://wlsguur.github.io/">Jinhyeok Choi</a><sup>*</sup> ·
  <a href="https://jaden-shin-1214.github.io/">Youngjin Shin</a> ·
  Jaeyeong Kim ·
  <a href="https://icetea-cv.github.io/">JongMin Lee</a> ·
  Seungryong Kim<sup>&dagger;</sup>
</p>

<p align="center">
  KAIST AI<br>
  <sup>*</sup> Equal contribution &nbsp;·&nbsp; <sup>&dagger;</sup> Corresponding author
</p>

<p align="center">
  <a href="https://cvlab-kaist.github.io/MORPHOS/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
  <a href="#"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b" alt="arXiv"></a>
</p>

---

## Abstract

We present **MORPHOS**, an autoregressive 4D generative framework that produces dynamic 3D assets from video across **diverse representations — meshes, 3D Gaussians, and Radiance Fields**. We introduce **Temporal Structured Latents (T-SLAT)**, a unified 4D representation that jointly encodes geometry and appearance over time. With causal attention, MORPHOS conditions each frame on its preceding history, and a **temporal-structural augmentation** strategy mitigates error accumulation for robust long-horizon generation.

## Release Plan

We are preparing the following for public release. Stay tuned!

- [ ] Inference code
- [ ] Evaluation code
- [ ] Training code
- [ ] T-SLAT training data
- [ ] Pretrained model weights

## Citation

If you find our work useful, please consider citing:

```bibtex
@article{kwon2026morphos,
  title   = {MORPHOS: Autoregressive 4D Generation with Temporal Structured Latents},
  author  = {Kwon, Minkyung and Choi, Jinhyeok and Shin, Youngjin and
             Kim, Jaeyeong and Lee, JongMin and Kim, Seungryong},
  journal = {arXiv preprint},
  year    = {2026}
}
```
