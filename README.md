
# Models/Git Branches

`main` : Uses triplanes to predict $\lambda, a, b, s$ all view dependant except $s$, and modelled via spherical harmonics. We the use $a,b,s$ to sample a low and high resolution texture using mip-maping/LERP method and use the equation $c' = \lambda c + (1-\lambda)\Delta c (\cdot)$ where $\Delta c(a,b,s,I_{text}) \rightarrow  \text{Texture Sampling}$. This is our naive base model. Much to explore...

# Data Collection Notes

1. Pretain the splatfacto model (via nerfstudio) on the set of canonical images including those for the relighting videos
2. Extract the initial gaussian model + poses for the relighting cameras (both require ns-generate command within nerfstudio)
3. Generate novel view camera path with nerfstuio
4. Load into current model...



