import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

#theta_snapshots - iterations of regression to animate
#feature_index - zmienna po ktorej animujemy
#ogolem pewnie bedzie trzeba to zmienic, ale cyborg wyplul cos tam dostosowalem, ogarniemy jak beda juz algosy
def animate_regression(X, y, theta_snapshots, feature_index, title): 
    x_vals = np.linspace(X[:, feature_index].min(), X[:, feature_index].max(), 100).reshape(-1, 1)

    X_vis = np.zeros((100, X.shape[1]))
    X_vis[:, 0] = 1  # bias
    X_vis[:, feature_index] = x_vals[:, 0]  # tylko ta jedna cecha się zmienia

    fig, ax = plt.subplots()
    ax.scatter(X[:, feature_index], y, color='blue', alpha=0.4, label='Data')
    line, = ax.plot([], [], color='red')
    # ax.set_xlabel('Surface Area (scaled)')
    # ax.set_ylabel('Cooling Load')
    # ax.set_title('Gradient Descent wpływ jednej cechy')
    ax.legend()

    def update(frame):
        theta = theta_snapshots[frame]
        y_vals = X_vis @ theta
        line.set_data(x_vals[:, 0], y_vals)
        ax.set_title(f'Iteracja: {frame * 100}')
        return line,

    anim = FuncAnimation(fig, update, frames=len(theta_snapshots), interval=300, blit=True)

    # To save as gif
    wr = animation.PillowWriter(fps=15, bitrate=1800)
    anim.save(f'animation_{title}.gif', writer=wr)

    plt.show()