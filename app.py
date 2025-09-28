# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz

st.set_page_config(page_title="Beam Analysis Tool", page_icon="📐", layout="centered")

st.title("📐 Beam / Bridge Analysis Tool")
st.write(
    "Interactive app to calculate reactions, shear force, bending moment, and deflection "
    "for a simply supported beam. Designed for learning and demonstration."
)

# -----------------------
# Sidebar: Inputs
# -----------------------
st.sidebar.header("Beam properties")
L = st.sidebar.number_input("Beam length L (m)", min_value=0.1, value=6.0, step=0.5)
E_gpa = st.sidebar.number_input("Young's modulus E (GPa)", min_value=0.1, value=200.0, step=1.0)
E = E_gpa * 1e9
I = st.sidebar.number_input("Moment of inertia I (m⁴)", min_value=1e-9, value=5e-5, step=1e-6, format="%.8f")

st.sidebar.header("Loading")
load_case = st.sidebar.selectbox("Load type", ["Point Load", "Uniformly Distributed Load (UDL)"])
if load_case == "Point Load":
    P_kn = st.sidebar.number_input("Point load P (kN)", min_value=0.0, value=10.0, step=1.0)
    P = P_kn * 1000.0
    a = st.sidebar.slider("Distance a from left support (m)", min_value=0.0, max_value=float(L), value=float(L/2))
    # allow an optional second point load for demonstration
    add_second = st.sidebar.checkbox("Add second point load (optional)", value=False)
    if add_second:
        P2_kn = st.sidebar.number_input("Second load P2 (kN)", min_value=0.0, value=5.0, step=1.0)
        P2 = P2_kn * 1000.0
        a2 = st.sidebar.slider("Distance a2 from left support (m)", min_value=0.0, max_value=float(L), value=float(L*3/4))
    else:
        P2 = 0.0
        a2 = None
else:
    w_knm = st.sidebar.number_input("UDL intensity w (kN/m)", min_value=0.0, value=5.0, step=0.5)
    w = w_knm * 1000.0

st.sidebar.header("Display options")
npts = st.sidebar.slider("Plot resolution (points)", 100, 2000, 500)
show_values = st.sidebar.checkbox("Show numeric reactions & maxs", value=True)

# -----------------------
# Domain
# -----------------------
x = np.linspace(0, L, npts)

# -----------------------
# Build Shear V(x) and Moment M(x)
# -----------------------
V = np.zeros_like(x)
M = np.zeros_like(x)

if load_case == "Point Load":
    # compute reactions for single point load + optional second point load
    # reactions R1, R2 found by equilibrium: R1 + R2 = P + P2 ; sum moments about left: R2*L = P*a + P2*a2
    total_load = P + P2
    if (P2 and a2 is not None) or (P2 == 0.0):
        moment_about_left = P * a + (P2 * a2 if P2 else 0.0)
        R2 = moment_about_left / L
        R1 = total_load - R2
    else:
        R1 = P * (L - a) / L
        R2 = P - R1

    # Shear: start with left reaction, subtract point loads when x >= position
    V += R1
    if P2:
        V = np.where(x < a, V, V - P)
        V = np.where(x < a2, V, V - P2)
    else:
        V = np.where(x < a, V, V - P)

    # Moment: integrate shear (piecewise expressions via trapezoid)
    # We'll compute moment by integrating shear numerically so it matches shear plot.
    M = cumtrapz(V, x, initial=0.0)

else:
    # UDL case: reactions equal w*L/2
    R1 = R2 = w * L / 2.0
    V = R1 - w * x
    M = cumtrapz(V, x, initial=0.0)

# -----------------------
# Deflection: solve y'' = M(x) / (E*I)
# We'll integrate twice and apply boundary conditions y(0)=0, y(L)=0 to find integration constants.
# -----------------------
# integrate M/EI to get slope
slope = cumtrapz(M / (E * I), x, initial=0.0)  # slope(x) = integral_0^x M/EI dx + C1
# integrate slope to get deflection (with unknown C1,C2)
y_unnormalized = cumtrapz(slope, x, initial=0.0)  # y(x) = integral_0^x slope dx + C1*x + C2

# Enforce boundary conditions:
# y(0) = C2 = 0  => C2 = 0
# y(L) = y_unnormalized[-1] + C1*L = 0 => C1 = -y_unnormalized[-1] / L
C1 = -y_unnormalized[-1] / L
y = y_unnormalized + C1 * x

# max deflection (m), take absolute max
max_deflection = np.min(y) if np.min(y) < 0 else np.max(y)
max_deflection_abs = np.max(np.abs(y))

# -----------------------
# Display results
# -----------------------
st.subheader("Results (support reactions & max deflection)")
st.write(f"- Reaction at left support (A): **{R1:.2f} N**")
st.write(f"- Reaction at right support (B): **{R2:.2f} N**")
st.write(f"- Maximum deflection (absolute): **{max_deflection_abs:.6e} m**")

# show where max deflection occurs
imax = np.argmax(np.abs(y))
st.write(f"- Max deflection location: x = **{x[imax]:.3f} m**")

# Optionally show shear & moment numeric extremes
if show_values:
    st.write(f"- Max bending moment: **{np.max(M):.2f} N·m**  |  Min bending moment: **{np.min(M):.2f} N·m**")
    st.write(f"- Max shear: **{np.max(V):.2f} N**  |  Min shear: **{np.min(V):.2f} N**")

# -----------------------
# Plotting
# -----------------------
fig, axes = plt.subplots(3, 1, figsize=(7, 8), constrained_layout=True)

axes[0].plot(x, V)
axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].set_title("Shear Force V(x)")
axes[0].set_ylabel("V (N)")
axes[0].grid(True)

axes[1].plot(x, M)
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_title("Bending Moment M(x)")
axes[1].set_ylabel("M (N·m)")
axes[1].grid(True)

axes[2].plot(x, y)
axes[2].axhline(0, color="black", linewidth=0.8)
axes[2].set_title("Deflection y(x)")
axes[2].set_xlabel("x (m)")
axes[2].set_ylabel("y (m)")
axes[2].grid(True)

st.pyplot(fig)

st.markdown("---")
st.caption("Developed by Manasa — Beam/Bridge Analysis Tool. For learning/demo purposes. License: MIT.")
