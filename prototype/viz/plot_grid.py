

def plot_grid(ax, nb_rows, nb_cols):
    grid = []

    for i in range(nb_rows):
        for j in range(nb_rows):
            grid.append([i, j])

    for pt in grid:
        ax.plot(pt[0], pt[1], marker="o")
