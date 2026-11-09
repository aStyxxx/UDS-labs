"""
Лабораторна робота №2
Тема: Моделювання задачі керування лінійною нестаціонарною системою
Динамічна система:
    dx/dt = 2*t*x + u, t in [0, T], T > 0
Крайові умови:
    x(0) = x0, x(T) = y0

Чотири класи керування:
    1. Постійне: u(t) = c
    2. Програмне лінійне: u(t) = c * t
    3. Зворотний зв'язок: u(x) = c * x (за умови x0 * y0 > 0)
    4. Кусково-постійне: u(t) = c1 при t in [0, t1], u(t) = c2 при t in (t1, T]
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# =====================================================================
# 1. БЛОК ВХІДНИХ ПАРАМЕТРІВ ЗАДАЧІ
# =====================================================================
x0 = 1.0       # Початковий стан системи в момент t = 0
y0 = 4.0       # Цільовий кінцевий стан системи в момент t = T
T = 1.0        # Кінцевий момент часу (T > 0)
t1 = 0.5       # Точка перемикання для кусково-постійного керування (0 < t1 < T)
c1 = -1.0      # Значення керування на інтервалі [0, t1] для п. 4

# Перевірка коректності вхідних параметрів
if T <= 0:
    raise ValueError(f"Помилка: параметр T має бути строго більшим за 0. Отримано T = {T}")
if not (0 < t1 < T):
    raise ValueError(f"Помилка: момент перемикання t1 має задовольняти умову 0 < t1 < T. Отримано t1 = {t1}, T = {T}")
if x0 * y0 <= 0:
    raise ValueError(
        f"Помилка: для побудови керування зі зворотним зв'язком (п. 3) необхідна умова x0 * y0 > 0!\n"
        f"Поточні значення: x0 = {x0}, y0 = {y0} (добуток {x0 * y0} <= 0)."
    )

# =====================================================================
# 2. ДОПОМІЖНІ ФУНКЦІЇ ДЛЯ ІНТЕГРАЛА ГАУССА
# =====================================================================
def I(a, b):
    """
    Точне аналітичне обчислення інтеграла Гаусса:
        I(a, b) = \\int_{a}^{b} exp(-s^2) ds = (sqrt(pi)/2) * [erf(b) - erf(a)]
    Підтримує як окремі числа, так і масиви NumPy.
    """
    return (np.sqrt(np.pi) / 2.0) * (erf(b) - erf(a))

# =====================================================================
# 3. АНАЛІТИЧНИЙ РОЗРАХУНОК КОНСТАНТ КЕРУВАННЯ
# =====================================================================
# 1. Постійне керування: u(t) = c
#    c = (y0 * exp(-T^2) - x0) / I(0, T)
c_const = (y0 * np.exp(-T**2) - x0) / I(0.0, T)

# 2. Програмне лінійне керування: u(t) = c * t
#    c = 2 * (y0 - x0 * exp(T^2)) / (exp(T^2) - 1)
c_lin = 2.0 * (y0 - x0 * np.exp(T**2)) / (np.exp(T**2) - 1.0)

# 3. Керування зі зворотним зв'язком: u(x) = c * x
#    c = (1 / T) * ln(y0 / x0) - T
c_fb = (1.0 / T) * np.log(y0 / x0) - T

# 4. Кусково-постійне керування:
#    c2 = (y0 * exp(-T^2) - x0 - c1 * I(0, t1)) / I(t1, T)
if np.isclose(c1, c_const):
    print(f"[Зауваження] Обране значення c1 = {c1:.4f} збігається з c постійного керування.")

c2_piecewise = (y0 * np.exp(-T**2) - x0 - c1 * I(0.0, t1)) / I(t1, T)

# =====================================================================
# 4. ОБЧИСЛЕННЯ ТРАЄКТОРІЙ ТА КЕРУВАНЬ НА СІТЦІ ЧАСУ
# =====================================================================
num_points = 500
t = np.linspace(0.0, T, num_points)

# 1. Постійне керування:
#    x(t) = exp(t^2) * (x0 + c * I(0, t))
#    u(t) = c
x_const = np.exp(t**2) * (x0 + c_const * I(0.0, t))
u_const = np.full_like(t, c_const)

# 2. Програмне лінійне керування:
#    x(t) = x0 * exp(t^2) + (c / 2) * (exp(t^2) - 1)
#    u(t) = c * t
x_lin = x0 * np.exp(t**2) + (c_lin / 2.0) * (np.exp(t**2) - 1.0)
u_lin = c_lin * t

# 3. Керування зі зворотним зв'язком:
#    x(t) = x0 * exp(t^2 + c * t)
#    u(t) = c * x(t)
x_fb = x0 * np.exp(t**2 + c_fb * t)
u_fb = c_fb * x_fb

# 4. Кусково-постійне керування:
#    При t in [0, t1]: x(t) = exp(t^2) * (x0 + c1 * I(0, t))
#    При t in (t1, T]: x(t) = exp(t^2) * (x0 + c1 * I(0, t1) + c2 * I(t1, t))
x_piecewise = np.where(
    t <= t1,
    np.exp(t**2) * (x0 + c1 * I(0.0, t)),
    np.exp(t**2) * (x0 + c1 * I(0.0, t1) + c2_piecewise * I(t1, t))
)
u_piecewise = np.where(t <= t1, c1, c2_piecewise)

# Значення стану в точці перемикання t1
x_at_t1 = np.exp(t1**2) * (x0 + c1 * I(0.0, t1))

# =====================================================================
# 5. КОНСОЛЬНЕ ВИВЕДЕННЯ ЗНАЧЕНЬ КОНСТАНТ ТА ПЕРЕВІРКА ПОХИБОК
# =====================================================================
err_const = abs(x_const[-1] - y0)
err_lin = abs(x_lin[-1] - y0)
err_fb = abs(x_fb[-1] - y0)
err_piecewise = abs(x_piecewise[-1] - y0)

print("=" * 80, flush=True)
print("  МОДЕЛЮВАННЯ ЗАДАЧІ КЕРУВАННЯ ЛІНІЙНОЮ СИСТЕМОЮ (ЛАБОРАТОРНА РОБОТА 2)", flush=True)
print("=" * 80, flush=True)
print(f"Параметри: x0 = {x0:.3f}, y0 = {y0:.3f}, T = {T:.3f}, t1 = {t1:.3f}, c1 = {c1:.3f}", flush=True)
print("-" * 80, flush=True)
print(f"1. Постійне керування:            c  = {c_const:10.6f} | x(T) = {x_const[-1]:.6f} | Похибка = {err_const:.2e}", flush=True)
print(f"2. Програмне лінійне керування:   c  = {c_lin:10.6f} | x(T) = {x_lin[-1]:.6f} | Похибка = {err_lin:.2e}", flush=True)
print(f"3. Керування зі зворотним зв'язком:c  = {c_fb:10.6f} | x(T) = {x_fb[-1]:.6f} | Похибка = {err_fb:.2e}", flush=True)
print(f"4. Кусково-постійне керування:    c1 = {c1:10.6f} | x(t1) = {x_at_t1:.6f}", flush=True)
print(f"                                  c2 = {c2_piecewise:10.6f} | x(T)  = {x_piecewise[-1]:.6f} | Похибка = {err_piecewise:.2e}", flush=True)
print("=" * 80, flush=True)

# =====================================================================
# 6. ВІЗУАЛІЗАЦІЯ ТРАЄКТОРІЙ ТА КЕРУВАНЬ (2 ОКРЕМІ ГРАФІКИ)
# =====================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))

# Колірна палітра та стилі ліній
colors = {
    'const': '#1f77b4',     # Синій
    'lin': '#ff7f0e',       # Помаранчевий
    'fb': '#2ca02c',        # Зелений
    'piece': '#9467bd'      # Фіолетовий
}

# ---------------------------------------------------------------------
# ГРАФІК 1: Траєкторії стану x(t)
# ---------------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(8.5, 6), dpi=300)
fig1.patch.set_facecolor('#ffffff')
ax1.set_facecolor('#fafbfc')

ax1.plot(t, x_const, label=f"1. Постійне (c = {c_const:.2f})", color=colors['const'], linestyle='-', linewidth=2.2)
ax1.plot(t, x_lin, label=f"2. Лінійне (c = {c_lin:.2f})", color=colors['lin'], linestyle='--', linewidth=2.2)
ax1.plot(t, x_fb, label=f"3. Зворотний зв'язок (c = {c_fb:.2f})", color=colors['fb'], linestyle='-.', linewidth=2.2)
ax1.plot(t, x_piecewise, label=f"4. Кусково-постійне (c₁ = {c1:.2f}, c₂ = {c2_piecewise:.2f})", color=colors['piece'], linestyle=':', linewidth=2.5)

# Позначення характерних точок
ax1.plot(0, x0, marker='o', markersize=8, color='#004085', zorder=5, label=f"Початок (0, {x0:.1f})")
ax1.plot(T, y0, marker='*', markersize=13, color='#721c24', zorder=5, label=f"Ціль (T, {y0:.1f})")
ax1.plot(t1, x_at_t1, marker='D', markersize=7, color='#6f42c1', zorder=5, label=f"Перемикання t₁ ({t1:.2f}, {x_at_t1:.2f})")

ax1.set_title("Фазові траєкторії стану системи $x(t)$", fontsize=13, fontweight='bold', pad=12)
ax1.set_xlabel("Час $t$", fontsize=11)
ax1.set_ylabel("Стан системи $x(t)$", fontsize=11)
ax1.grid(True, linestyle='--', alpha=0.3)
ax1.legend(loc='best', fontsize=9.5, framealpha=0.92)
fig1.tight_layout()

file_traj = os.path.join(script_dir, "state_trajectories.png")
fig1.savefig(file_traj, dpi=300, bbox_inches='tight')
print(f"[Успіх] Графік траєкторій стану збережено у: {file_traj}", flush=True)

# ---------------------------------------------------------------------
# ГРАФІК 2: Функції керування u(t)
# ---------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(8.5, 6), dpi=300)
fig2.patch.set_facecolor('#ffffff')
ax2.set_facecolor('#fafbfc')

ax2.plot(t, u_const, label=f"1. Постійне (c = {c_const:.2f})", color=colors['const'], linestyle='-', linewidth=2.2)
ax2.plot(t, u_lin, label=f"2. Лінійне (u = {c_lin:.2f}·t)", color=colors['lin'], linestyle='--', linewidth=2.2)
ax2.plot(t, u_fb, label=f"3. Зворотний зв'язок (u = {c_fb:.2f}·x)", color=colors['fb'], linestyle='-.', linewidth=2.2)

# Для кусково-постійного керування будуємо сходинку з чіткою вертикальною лінією
t_step = np.array([0.0, t1, t1, T])
u_step = np.array([c1, c1, c2_piecewise, c2_piecewise])
ax2.plot(t_step, u_step, label=f"4. Кусково-постійне (c₁ = {c1:.2f}, c₂ = {c2_piecewise:.2f})", color=colors['piece'], linestyle=':', linewidth=2.5)

# Вертикальна лінія для позначення моменту перемикання t1
ax2.axvline(t1, color='#6c757d', linestyle='--', linewidth=1.0, alpha=0.6)
ax2.plot([t1, t1], [c1, c2_piecewise], marker='o', markersize=5, linestyle='', color=colors['piece'])

ax2.set_title("Функції керування $u(t)$", fontsize=13, fontweight='bold', pad=12)
ax2.set_xlabel("Час $t$", fontsize=11)
ax2.set_ylabel("Керування $u(t)$", fontsize=11)
ax2.grid(True, linestyle='--', alpha=0.3)
ax2.legend(loc='best', fontsize=9.5, framealpha=0.92)
fig2.tight_layout()

file_ctrl = os.path.join(script_dir, "control_functions.png")
fig2.savefig(file_ctrl, dpi=300, bbox_inches='tight')
print(f"[Успіх] Графік функцій керування збережено у: {file_ctrl}", flush=True)

# Відображення вікон графіків
plt.show()

