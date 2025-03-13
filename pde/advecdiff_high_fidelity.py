import numpy as np
import matplotlib.pyplot as plt
from dolfin import *

# Define mesh and function space28
Nx = 512
mesh = UnitSquareMesh(Nx, Nx)  # High resolution
V = FunctionSpace(mesh, "CG", 2)  # Quadratic elements (P2)

# Define problem parameters
D = Constant(1e-3)  # Diffusion coefficient
b = Constant((1.0, 0.0))  # Convection vector

# Define boundary condition (Dirichlet: u = 0 on ∂Ω)
g = Constant(0.0)
bc = DirichletBC(V, g, "on_boundary")

# Define source term f = 1
f = Constant(2.0)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)

# SUPG stabilization parameter
h = CellDiameter(mesh)
tau_SUPG = h / (2 * abs(b[0]) + 1e-10)  # Avoid division by zero

# Weak formulation with SUPG stabilization (fixing the sign for diffusion)
a = (D * inner(grad(u), grad(v)) + inner(b, grad(u)) * v + tau_SUPG * inner(b, grad(u)) * inner(b, grad(v))) * dx
L = (f * v + tau_SUPG * f * inner(b, grad(v))) * dx

# Solve problem
u_sol = Function(V)
solve(a == L, u_sol, bc)

# ✅ Interpolate onto a linear function space (P1) to match mesh vertices
V_linear = FunctionSpace(mesh, "CG", 1)
u_sol_linear = interpolate(u_sol, V_linear)

u_values = u_sol_linear.compute_vertex_values(mesh)
x, y = mesh.coordinates().T  # Extract (x, y) coordinates of mesh vertices

# # ✅ Extract triangular connectivity
# triangles = np.array([cell.entities(0) for cell in cells(mesh)])

# 📌 **Corrected Contour Plot using Tricontour**
plt.figure(figsize=(8, 6))
plt.contourf(x.reshape(Nx+1, Nx+1), y.reshape(Nx+1, Nx+1), u_values.reshape(Nx+1, Nx+1), levels=100, cmap="coolwarm")
plt.colorbar(label="Solution u")
plt.title("High-Fidelity FEM Solution (Convection-Diffusion)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Save solution for visualization in Paraview
# File("solution.pvd") << u_sol

# save high fidelity solution 

# save high fidelity solution
np.save("pde/u_values.npy", u_values)