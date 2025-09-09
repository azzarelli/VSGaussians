import numpy as np
import plotly.graph_objects as go

# Example: load your dict
cams = {
    "focals": [293.7, 293.7, 293.7, 293.7],
    "imsize": [[512,288]]*4,
    "cams2world": [
        [[-0.22010836, 0.61132306, 0.7601558, -1.08676577],
         [-0.33892196, 0.68279326, -0.64724469, 1.32122195],
         [-0.91470468, -0.40009743, 0.05690227, 0.55018514],
         [0., 0., 0., 1.]],
        [[0.01310995, -0.01104844, 0.99985313, -1.86468184],
         [-0.46655884, 0.88434756, 0.01588961, -0.04510819],
         [-0.88439322, -0.46669859, 0.00643897, 0.61029184],
         [0., 0., 0., 1.]],
        [[-0.0366444, 0.60274184, 0.79709452, -1.10154605],
         [-0.45095021, 0.70182848, -0.55143541, 1.01916254],
         [-0.89179671, -0.37965694, 0.24608842, -0.03157448],
         [0., 0., 0., 1.]],
        [[0.25262794, 0.59175229, 0.76551193, -1.02024126],
         [-0.59142554, 0.72060549, -0.36186153, 0.58435792],
         [-0.76576442, -0.36132693, 0.53202248, -0.70093787],
         [0., 0., 0., 1.]]
    ]
}

# Make into numpy arrays
cams2world = np.array(cams["cams2world"])  # shape (N,4,4)

# Extract camera centers (translation part)
centers = cams2world[:, :3, 3]

# Extract forward vectors (-Z axis of rotation)
forwards = -cams2world[:, :3, 2]  # because camera looks along -Z in its local frame

# --- Example: load your saved point cloud too ---
scene = np.load("pointcloud.npy", allow_pickle=True).item()
pts = scene["points"]   # (N,3)
col = scene["colors"]   # (N,3)

# --- Plot ---
fig = go.Figure()

# Point cloud
fig.add_trace(go.Scatter3d(
    x=pts[:,0], y=pts[:,1], z=pts[:,2],
    mode="markers",
    marker=dict(
        size=2,
        color=["rgb({},{},{})".format(r,g,b) for r,g,b in col],
        opacity=0.8
    ),
    name="Points"
))

# Cameras as cones
for C, v in zip(centers, forwards):
    fig.add_trace(go.Cone(
        x=[C[0]], y=[C[1]], z=[C[2]],
        u=[v[0]], v=[v[1]], w=[v[2]],
        sizemode="absolute",
        sizeref=0.2,   # adjust for bigger/smaller cones
        anchor="tip",
        colorscale="Blues",
        showscale=False,
        name="Camera"
    ))

fig.update_layout(
    scene=dict(aspectmode="data"),
    margin=dict(l=0, r=0, t=0, b=0)
)

fig.show()
