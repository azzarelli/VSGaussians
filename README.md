
# Installation

This has been tested on NVidia RTX 3090 w/ py3.10 pt2.4 and cu11.8, and NVidia RTX 4090 w/ py3.10 pt2.4 and cu12.4

1. Create a conda environment with `conda create -n vsres python=3.10` (pytorch version requires compatibility with `gsplat`; I use py310)
2. Install pytorch (requires compatibility with `gsplat`; I use pt24)
3. Download `gsplat` either via `pip install gsplat` or with wheel (I used `gsplat-1.5.3+pt24cu124-cp310-cp310-linux_x86_64.whl`)
4. Run `pip install -r requirements.txt`
5. (Optional) Cry because that probably didn't work and the guy who created this repo also has no clue, so you choose not to start an issue because there's no way he's resolving it 



# Models/Git Branches

`main` : Uses triplanes to predict $\lambda, a, b, s$ all view dependant except $s$, and modelled via spherical harmonics. We the use $a,b,s$ to sample a low and high resolution texture using mip-maping/LERP method and use the equation $c' = \lambda c + (1-\lambda)\Delta c (\cdot)$ where $\Delta c(a,b,s,I_{text}) \rightarrow  \text{Texture Sampling}$. This is our naive base model. Much to explore...

`fully_explicit` : Models the spherical harmonics/1-channel parameters for $\lambda, a, b, s$ as per-gaussian points. Otherwise retains similarity to `main`

`fe_deformation` : I forget but maybe I changed from alphablend function to a deformation function

`no_opacity_param` : Remove the opacity variable and treat every point as having opacity=1. (not efficient implementation)

`canon_loss` : Use canonical images to train the canonical color and geometry components

# Data Collection Notes

1. Pretain the splatfacto model (via nerfstudio) on the set of canonical images including those for the relighting videos
2. Extract the initial gaussian model + poses for the relighting cameras (both require ns-generate command within nerfstudio)
3. Generate novel view camera path with nerfstuio
4. Load into current model...



