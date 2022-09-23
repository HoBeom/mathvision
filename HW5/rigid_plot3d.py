# Base code from https://matplotlib.org/2.1.2/gallery/animation/simple_3danim.html
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from rigid import RigidTransform
import torch

WITH_ANIMATION = True
RANDOM_POINT_NUM = 15
RANDOM_CENTER = 1
RANDOM_SCALE = 5
# Fixing random state for reproducibility
np.random.seed(19680801)


points_3d = torch.tensor(
    [
        [-0.500000, 0.000000, 2.121320],
        [0.500000, 0.000000, 2.121320],
        [0.500000, -0.707107, 2.828427],
    ],
    dtype=torch.float64,
)
points_3d_transformed = torch.tensor(
    [
        [1.363005, -0.427130, 2.339082],
        [1.748084, 0.437983, 2.017688],
        [2.636461, 0.184843, 2.400710],
    ],
    dtype=torch.float64,
)

transform = RigidTransform(points_3d, points_3d_transformed)


def Gen_RandPoint3D(number, scale):
    # Generate random points
    data = np.random.rand(number, 3) - 0.5
    data *= scale
    return data


# animate points 3d to points 3d transformed
def update_lines(num, dataLines, lines):
    for line, data in zip(lines, dataLines):
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(data[: num, :2].T)
        line.set_3d_properties(data[: num, 2])
    return lines

# Attaching 3D axis to the figure
fig = plt.figure()
ax = p3.Axes3D(fig)

ax.set_title("3D Test")

# plot points
ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], color="red")
ax.scatter(
    points_3d_transformed[:, 0],
    points_3d_transformed[:, 1],
    points_3d_transformed[:, 2],
    color="blue",
)
# draw lines points 3d to points 3d transformed
for i in range(points_3d.shape[0]):
    ax.plot(
        [points_3d[i, 0], points_3d_transformed[i, 0]],
        [points_3d[i, 1], points_3d_transformed[i, 1]],
        [points_3d[i, 2], points_3d_transformed[i, 2]],
        color="black",
    )

# Creating the Animation object
# random 3-D points
data = Gen_RandPoint3D(RANDOM_POINT_NUM, RANDOM_SCALE) + RANDOM_CENTER
data = torch.tensor(data, dtype=torch.float64)
data_transformed = transform(data)
print(data_transformed)

# plot points
ax.scatter(data[:, 0], data[:, 1], data[:, 2], color="green")
ax.scatter(
    data_transformed[:, 0],
    data_transformed[:, 1],
    data_transformed[:, 2],
    color="yellow",
)
# draw lines points 3d to points 3d transformed
for i in range(data.shape[0]):
    ax.plot(
        [data[i, 0], data_transformed[i, 0]],
        [data[i, 1], data_transformed[i, 1]],
        [data[i, 2], data_transformed[i, 2]],
        color="black",
        linestyle="dashed",
    )

if WITH_ANIMATION:
    # make lines data to data transformed
    data = torch.cat([points_3d, data], dim=0)
    data_transformed = torch.cat([points_3d_transformed, data_transformed], dim=0)
    lines = []
    for i in range(data.shape[0]):
        xline = torch.linspace(data[i, 0], data_transformed[i, 0], 25)
        yline = torch.linspace(data[i, 1], data_transformed[i, 1], 25)
        zline = torch.linspace(data[i, 2], data_transformed[i, 2], 25)
        lines.append(torch.stack([xline, yline, zline], dim=1))
    lines = torch.stack(lines)
    print(lines)

    line_ani = animation.FuncAnimation(
        fig,
        update_lines,
        25,
        fargs=(lines, ax.lines),
        interval=50,
        blit=False,
    )
    line_ani.save('lines.gif', writer='imagemagick')

plt.show()
# save ani gif
