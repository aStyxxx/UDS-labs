"""
Laboratory Work: Variant 13
dx/dt = 2x - y
dy/dt = 5x - 2y
Eigenvalues: lambda = +-i (Center equilibrium)
Analytical solution:
    x(t) = x0*cos(t) + (2*x0 - y0)*sin(t)
    y(t) = y0*cos(t) + (5*x0 - 2*y0)*sin(t)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def solve_ode(x0, y0, t):
    # Analytical solution via matrix exponential exp(A*t) = cos(t)*I + sin(t)*A
    x = x0 * np.cos(t) + (2 * x0 - y0) * np.sin(t)
    y = y0 * np.cos(t) + (5 * x0 - 2 * y0) * np.sin(t)
    return x, y


def generate_initial_points(radius=13.0, inner_radii=(3.25, 6.5, 9.75), n_boundary=24, n_inner=16):
    # Sample initial points from domain G0: boundary circle (R=13) and concentric inner circles
    theta_b = np.linspace(0, 2 * np.pi, n_boundary, endpoint=False)
    boundary_points = [(radius * np.cos(th), radius * np.sin(th)) for th in theta_b]

    inner_points = []
    for r in inner_radii:
        theta_i = np.linspace(0, 2 * np.pi, n_inner, endpoint=False)
        for th in theta_i:
            inner_points.append((r * np.cos(th), r * np.sin(th)))

    return boundary_points, inner_points


def save_figure(fig, filename):
    # Save high-resolution copy in both script directory and project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, filename)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')

    root_output_path = os.path.abspath(filename)
    if root_output_path != output_path:
        fig.savefig(root_output_path, dpi=300, bbox_inches='tight')


def plot_phase_portrait(boundary_points, inner_points, radius=13.0, T=10.0):
    t = np.linspace(0.0, T, 1000)

    fig, ax = plt.subplots(figsize=(7.5, 9.5), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#fafbfc')

    # Normalized velocity field (dx/dt, dy/dt); avoid 0/0 division at the origin
    x_grid = np.linspace(-36, 36, 23)
    y_grid = np.linspace(-74, 74, 25)
    X, Y = np.meshgrid(x_grid, y_grid)
    U = 2 * X - Y
    V = 5 * X - 2 * Y
    norm = np.hypot(U, V)
    norm_safe = np.where(norm == 0, 1.0, norm)
    U_norm = np.where(norm > 0, U / norm_safe, 0)
    V_norm = np.where(norm > 0, V / norm_safe, 0)
    ax.quiver(X, Y, U_norm, V_norm, color='#cbd5e1', alpha=0.55,
              angles='xy', scale_units='xy', scale=0.55, width=0.0028,
              headwidth=3.5, headlength=4, label='Векторне поле $(dx/dt, dy/dt)$')

    # Initial circular region G0 at t = 0
    initial_circle = Circle((0, 0), radius, color='#38bdf8', alpha=0.22,
                            label=f'Початкова область $G_0$ ($R = {int(radius)}$)')
    initial_circle_border = Circle((0, 0), radius, edgecolor='#0284c7', facecolor='none',
                                   linestyle='--', linewidth=1.8,
                                   label='Межа області $x_0^2 + y_0^2 = 13^2$')
    ax.add_patch(initial_circle)
    ax.add_patch(initial_circle_border)

    # Inner trajectories
    for idx, (x0, y0) in enumerate(inner_points):
        xi, yi = solve_ode(x0, y0, t)
        lbl = 'Внутрішні траєкторії' if idx == 0 else None
        ax.plot(xi, yi, color='#64748b', alpha=0.35, linewidth=0.9, label=lbl)

    # Boundary trajectories with flow direction arrows
    for idx, (x0, y0) in enumerate(boundary_points):
        xb, yb = solve_ode(x0, y0, t)
        lbl = 'Межові траєкторії' if idx == 0 else None
        ax.plot(xb, yb, color='#2563eb', alpha=0.75, linewidth=1.3, label=lbl)

        idx_arr = 100
        ax.annotate('', xy=(xb[idx_arr + 1], yb[idx_arr + 1]),
                    xytext=(xb[idx_arr], yb[idx_arr]),
                    arrowprops=dict(arrowstyle="->", color="#1d4ed8", lw=1.2))

    # Initial points (t=0) and terminal points (t=10) showing flow deformation
    all_x0 = [p[0] for p in boundary_points]
    all_y0 = [p[1] for p in boundary_points]
    ax.scatter(all_x0, all_y0, color='#16a34a', s=26, zorder=5,
               edgecolors='black', linewidths=0.5, label='Початок траєкторій ($t = 0$)')

    end_x, end_y = [], []
    for x0, y0 in boundary_points:
        xt, yt = solve_ode(x0, y0, T)
        end_x.append(xt)
        end_y.append(yt)

    # Deformed boundary contour at t = T (Liouville theorem: area is preserved)
    end_x_closed = end_x + [end_x[0]]
    end_y_closed = end_y + [end_y[0]]
    ax.plot(end_x_closed, end_y_closed, color='#dc2626', linestyle='--', linewidth=1.5,
            alpha=0.85, label=f'Деформована межа $G_{{{int(T)}}}$ ($t = {int(T)}$)')

    ax.scatter(end_x, end_y, color='#dc2626', s=26, marker='^', zorder=5,
               edgecolors='black', linewidths=0.5, label=f'Кінець траєкторій ($t = {int(T)}$)')

    # Equilibrium point (Center)
    ax.scatter([0], [0], color='#111827', s=50, marker='x', zorder=6,
               linewidths=2.2, label='Особлива точка $(0,0)$ [Центр]')

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-38, 38)
    ax.set_ylim(-76, 76)
    ax.axhline(0, color='#64748b', linewidth=0.8, linestyle='-', alpha=0.4)
    ax.axvline(0, color='#64748b', linewidth=0.8, linestyle='-', alpha=0.4)
    ax.grid(True, linestyle=':', alpha=0.6, color='#94a3b8')
    ax.set_xlabel('Координата $x$', fontsize=12, fontweight='bold', labelpad=8)
    ax.set_ylabel('Координата $y$', fontsize=12, fontweight='bold', labelpad=8)
    ax.set_title('Фазовий портрет системи\nПучок траєкторій (Центр, $\\lambda_{1,2} = \\pm i$)',
                 fontsize=14, fontweight='bold', pad=14)
    ax.legend(loc='upper left', fontsize=8.8, framealpha=0.92, facecolor='#ffffff', edgecolor='#cbd5e1')

    plt.tight_layout()
    save_figure(fig, 'phase_portrait.png')
    return fig


def plot_time_dynamics(T=10.0):
    t = np.linspace(0.0, T, 1000)

    fig, ax = plt.subplots(figsize=(10.5, 6.0), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#fafbfc')

    # Selected initial points to illustrate harmonic oscillations
    demo_points = [
        {'x0': 13.0, 'y0': 0.0, 'name': r'$(x_0=13, y_0=0)$', 'color': '#2563eb'},
        {'x0': 0.0, 'y0': 13.0, 'name': r'$(x_0=0, y_0=13)$', 'color': '#dc2626'},
        {'x0': 6.5, 'y0': 0.0, 'name': r'$(x_0=6.5, y_0=0)$', 'color': '#16a34a'}
    ]

    for item in demo_points:
        x_sol, y_sol = solve_ode(item['x0'], item['y0'], t)
        c = item['color']
        pt_name = item['name']
        ax.plot(t, x_sol, color=c, linestyle='-', linewidth=2.0,
                label=f'$x(t)$ для {pt_name}')
        ax.plot(t, y_sol, color=c, linestyle='--', linewidth=1.6, alpha=0.85,
                label=f'$y(t)$ для {pt_name}')

    # Mark oscillation period T = 2*pi
    t_period = 2 * np.pi
    ax.axvline(t_period, color='#7c3aed', linestyle=':', linewidth=1.8,
               label=rf'Період $T = 2\pi \approx {t_period:.2f}$')
    ax.text(t_period + 0.12, 45, rf'$T = 2\pi$', color='#7c3aed',
            fontweight='bold', fontsize=10.5)

    ax.axhline(0, color='#64748b', linewidth=0.8, linestyle='-', alpha=0.4)
    ax.grid(True, linestyle=':', alpha=0.6, color='#94a3b8')
    ax.set_xlim(0.0, T)
    ax.set_xlabel('Час $t$', fontsize=12, fontweight='bold', labelpad=8)
    ax.set_ylabel('Значення координат $x(t), y(t)$', fontsize=12, fontweight='bold', labelpad=8)
    ax.set_title('Динаміка координат у часі\nГармонічні коливання на інтервалі $t \\in [0, 10]$',
                 fontsize=14, fontweight='bold', pad=14)
    ax.legend(loc='upper right', fontsize=8.8, framealpha=0.92, facecolor='#ffffff',
              edgecolor='#cbd5e1', ncol=1)

    plt.tight_layout()
    save_figure(fig, 'time_dynamics.png')
    return fig


def main():
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 1.0

    R = 13.0
    T = 10.0

    boundary_points, inner_points = generate_initial_points(radius=R)
    plot_phase_portrait(boundary_points, inner_points, radius=R, T=T)
    plot_time_dynamics(T=T)

    # Display plots if interactive GUI backend is available
    interactive_backends = {'tkagg', 'qtagg', 'qt5agg', 'wxagg', 'macosx'}
    current_backend = plt.get_backend().lower()
    if current_backend in interactive_backends or 'agg' not in current_backend:
        plt.show()

    plt.close('all')


if __name__ == '__main__':
    main()
