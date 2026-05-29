

pinn_keywords = [
    "Physics Informed",
    "Physics Informed Loss",
    "Theory-Trained Neural Networks",
    "TTN",
    "Physics Constrained",
    "Neural PDE solvers",
    "Neural PDE",
    "Neural differential equation solvers",
    "Neural Field",
]

nn_keywords = [
    "Neural Network",
    "Neural Net",
    "Neural Field",
    "Artificial Neural Network",
    "Feed-forward Neural Network",
    "Convolutional Neural Network",
    "U-Net",
    "Deep Learning",
    "Machine Learning",
    "Artificial Intelligence",
    "Optimisation-based learning",
    "Optimization-based learning",
    "Multi-layer perceptrons",
    "Model fitting",
]

pde_keywords = [
    "Partial Differential Equation",
    "PDE Solving",
    "PDE",
    "Constraint",
]

# Make permutations and combinations of these
pinn_search_segment = " OR ".join([f'\"{kw}\"' for kw in pinn_keywords])
nn_search_segment = " OR ".join([f'\"{kw}\"' for kw in nn_keywords])
pde_search_segment = " OR ".join([f'\"{kw}\"' for kw in pde_keywords])

search_term = f"({pinn_search_segment}) AND ({nn_search_segment}) AND ({pde_search_segment}) AND (\"Review\")"

print(search_term)