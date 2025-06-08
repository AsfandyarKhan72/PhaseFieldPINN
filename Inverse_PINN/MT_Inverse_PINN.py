# ---------------------------------------------------------------------------------------------------------------
# Importing Libraries and Configuration Setting
# ---------------------------------------------------------------------------------------------------------------
import os
os.environ["DDE_BACKEND"] = "pytorch"
import torch
import deepxde as dde
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deepxde.callbacks import Callback
from deepxde.icbc import PointSetBC
from deepxde.icbc import PeriodicBC
torch.cuda.empty_cache()

dde.config.set_random_seed(2025)
dde.config.set_default_float("float32")

# ---------------------------------------------------------------------------------------------------------------
# Defining Problem Parameters
# ---------------------------------------------------------------------------------------------------------------

# Spatial domain of the problem
Start_Length = 0.0
End_Length = 1.0

# Time domain of the problem
T_start = 0.0
T_end = 1.0

# Trainable Gradient energy coefficient
Beta_trainable = dde.Variable(1.0) # Initial guess value for Beta

# Chemical driving force
DelF = 0.036
# Kinetic coefficient
L = 10
# Elastic constants
C11 = 264.24
C12 = 115.38
C44 = 153.86
# Energy density coefficients
a, b, c = 0.2, -12.6, 12.4  


# ---------------------------------------------------------------------------------------------------------------
# Scaling Functions
# ---------------------------------------------------------------------------------------------------------------

# Output scaling factors
u_scale = 1e-3 # Scaling factor for u1 and u2
eta_scale = 1e-1 # Scaling factor for eta1 and eta2
def transform_func(X, Y):
    # Extracting individual outputs
    u1_raw, u2_raw, eta1_raw, eta2_raw = Y[:, 0:1], Y[:, 1:2], Y[:, 2:3], Y[:, 3:4]
    # Applying scaling
    u1_scaled = u1_raw * u_scale 
    u2_scaled = u2_raw * u_scale
    eta1_scaled = eta1_raw * eta_scale
    eta2_scaled = eta2_raw * eta_scale
    # Transformed outputs
    return torch.cat([u1_scaled, u2_scaled, eta1_scaled, eta2_scaled], dim=1)

# ---------------------------------------------------------------------------------------------------------------
# Defining the PDE Function
# ---------------------------------------------------------------------------------------------------------------
def pde(x, y):

    # Outputs of the neural network, u1, u2, eta1, eta2:  u1, and u2 are the displacement components, and eta1 and eta2 are the order parameters.
    u1, u2, eta1, eta2 = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4]

    u1_x = dde.grad.jacobian(y, x, i=0, j=0) # du1/dx
    u1_y = dde.grad.jacobian(y, x, i=0, j=1) # du1/dy

    u2_x = dde.grad.jacobian(y, x, i=1, j=0) # du2/dx
    u2_y = dde.grad.jacobian(y, x, i=1, j=1) # du2/dy

    eta1_x = dde.grad.jacobian(y, x, i=2, j=0) # deta1/dx
    eta1_y = dde.grad.jacobian(y, x, i=2, j=1) # deta1/dy

    eta2_x = dde.grad.jacobian(y, x, i=3, j=0) #  deta2/dx
    eta2_y = dde.grad.jacobian(y, x, i=3, j=1) #  deta2/dy

    eta1_t = dde.grad.jacobian(y, x, i=2, j=2) # deta1/dt
    eta2_t = dde.grad.jacobian(y, x, i=3, j=2) # deta2/dt

    # Stress components
    sigma11 = (7.443 * (eta2 ** 2 - eta1 ** 2) + C11 * u1_x + C12 * u2_y)
    sigma12 = (C44 * (u2_x + u1_y))
    sigma21 = sigma12
    sigma22 = (7.443 * (eta1 ** 2 - eta2 ** 2) + C12 * u1_x + C11 * u2_y)

    # Stress derivatives
    sigma11_x = dde.grad.jacobian(sigma11, x, i=0, j=0) # dsigma11/dx
    sigma12_y = dde.grad.jacobian(sigma12, x, i=0, j=1) # dsigma12/dy

    sigma21_x = dde.grad.jacobian(sigma21, x, i=0, j=0) # dsigma21/dx
    sigma22_y = dde.grad.jacobian(sigma22, x, i=0, j=1) # dsigma22/dy

    # Mechanical equilibrium equations (Eq. 1) and (Eq. 2):
    eq1 = (sigma11_x + sigma12_y)
    eq2 = (sigma21_x + sigma22_y)

    # Coefficient terms for TDGL equations (Eq. 3) and (Eq. 4):
    c_eta1_x = -(L * Beta_trainable) * eta1_x 
    c_eta1_y = -(L * Beta_trainable) * eta1_y 
    c_eta2_x = -(L * Beta_trainable) * eta2_x 
    c_eta2_y = -(L * Beta_trainable) * eta2_y 

    # Derivatives of above coefficients
    eta1_div = dde.grad.jacobian(c_eta1_x, x, i=0, j=0) + dde.grad.jacobian(c_eta1_y, x, i=0, j=1) # d(c_eta1_x)/dx + d(c_eta1_y)/dy
    eta2_div = dde.grad.jacobian(c_eta2_x, x, i=0, j=0) + dde.grad.jacobian(c_eta2_y, x, i=0, j=1) # d(c_eta2_x)/dx + d(c_eta2_y)/dy

    # Source terms for TDGL (Eq. 3) and (Eq. 4):
    f1 = L * ((1.4886 * eta1 * (-eta1 ** 2 + eta2 ** 2 + 10 * u1_x - 10 * u2_y)) - (DelF * (a * eta1 + b * eta1 ** 2 + c * eta1 * (eta1 ** 2 + eta2 ** 2))))
    f2 = -L * ((1.4886 * eta2 * (-eta1 ** 2 + eta2 ** 2 + 10 * u1_x - 10 * u2_y)) + (DelF * (a * eta2 + b * eta2 ** 2 + c * eta2 * (eta1 ** 2 + eta2 ** 2))))

    # TDGL equations (Eq. 3) and (Eq. 4):
    equ3 = eta1_t + eta1_div - f1  
    equ4 = eta2_t + eta2_div - f2

    return [eq1, eq2, equ3, equ4]


# ---------------------------------------------------------------------------------------------------------------
# Defining Geometry and Time Domain
# ---------------------------------------------------------------------------------------------------------------

geom = dde.geometry.Rectangle(xmin=[Start_Length, Start_Length], xmax=[End_Length, End_Length]) # Spatial domain
timedomain = dde.geometry.TimeDomain(T_start, T_end) # Time domain
geomtime = dde.geometry.GeometryXTime(geom, timedomain) # Geometry and time domain

# ---------------------------------------------------------------------------------------------------------------
# Boundary Conditions
# --------------------------------------------------------------------------------------------------------------

# Boundary checking in x (x1) directions
def boundary_x(X, on_boundary):
    return on_boundary and (np.isclose(X[0], 0) or np.isclose(X[0], 1))

# Boundary checking in y (x2) directions
def boundary_y(X, on_boundary):
    return on_boundary and (np.isclose(X[1], 0) or np.isclose(X[1], 1))


# Periodic BCs in x (x1) direction
bc_eta1_x = PeriodicBC(geomtime, component_x=0, on_boundary=boundary_x, derivative_order=0, component=2)  # Function periodicity
bc_flux_eta1_x = PeriodicBC(geomtime, component_x=0, on_boundary=boundary_x, derivative_order=1, component=2)  # Gradient periodicity

bc_eta2_x = PeriodicBC(geomtime, component_x=0, on_boundary=boundary_x, derivative_order=0, component=3)  # Function periodicity
bc_flux_eta2_x = PeriodicBC(geomtime, component_x=0, on_boundary=boundary_x, derivative_order=1, component=3)  # Gradient periodicityy

# Periodic BCs in y (x2) direction
bc_eta1_y = PeriodicBC(geomtime, component_x=1, on_boundary=boundary_y, derivative_order=0, component=2)  # Function periodicity
bc_flux_eta1_y = PeriodicBC(geomtime, component_x=1, on_boundary=boundary_y, derivative_order=1, component=2)  # Gradient periodicity

bc_eta2_y = PeriodicBC(geomtime, component_x=1, on_boundary=boundary_y, derivative_order=0, component=3)  # Function periodicity
bc_flux_eta2_y = PeriodicBC(geomtime, component_x=1, on_boundary=boundary_y, derivative_order=1, component=3)  # Gradient periodicity

# ---------------------------------------------------------------------------------------------------------------
# Initial Conditions and Observed Data
# ---------------------------------------------------------------------------------------------------------------

# Initial condition data for t = T_start
ic_data = np.loadtxt(f"IC_inverse_PINN.txt", delimiter=',')
xy_ic = ic_data[:, 0:2]  # x, y coordinates
eta1_ic = ic_data[:, 2:3]  # eta1 values
eta2_ic = ic_data[:, 3:4]  # eta2 values
xyt_ic = np.hstack((xy_ic, np.zeros((xy_ic.shape[0], 1)))) 
ic_eta1 = PointSetBC(xyt_ic, eta1_ic, component=2)
ic_eta2 = PointSetBC(xyt_ic, eta2_ic, component=3)

# Observe reference data for t = T_end (steady state)
observe_data =np.loadtxt(f"Obs_data_101x101.txt", delimiter=',')
xy_observ = observe_data[:, 0:2]  # x, y coordinates
eta1_observ = observe_data[:, 2:3]  # eta1 values
eta2_observ = observe_data[:, 3:4]  # eta2 values
xyt_observ = np.hstack((xy_observ, np.full((xy_observ.shape[0], 1), T_end)))  
observ_eta1 = PointSetBC(xyt_observ, eta1_observ, component=2)
observ_eta2 = PointSetBC(xyt_observ, eta2_observ, component=3)

# Anchor points
anchor_points = np.vstack((xyt_ic, xyt_observ)) # Anchor points for the initial condition and observe data


# ---------------------------------------------------------------------------------------------------------------
# Storing the component losses in a file
# ---------------------------------------------------------------------------------------------------------------

file_path = "Losses_FilePath"

# Cleaning up existing file if there
if os.path.exists(file_path):
    os.remove(file_path)
    print(f"Removed existing file: {file_path}")

# Callback Class for tracking losses
class SimpleLossTrackingCallback(Callback):
    def __init__(self, every_n_epochs=1000, file_path=file_path):
        super(SimpleLossTrackingCallback, self).__init__()
        self.every_n_epochs = every_n_epochs
        self.file_path = file_path
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        if not os.path.exists(self.file_path) or os.stat(self.file_path).st_size == 0:
            with open(self.file_path, "w") as f:
                f.write("Epoch,PDE1 Loss,PDE2 Loss,PDE3 Loss,PDE4 Loss,"
                        "BC1 Loss,BC2 Loss,BC3 Loss,BC4 Loss,BC5 Loss,BC6 Loss,BC7 Loss,BC8 Loss,"
                        "IC1 Loss,IC2 Loss,Observed_eta1_data_loss,Observed_eta2_data_loss\n")
    def on_epoch_end(self):
        step = self.model.train_state.step
        if step % self.every_n_epochs == 0 or step == 1:
            current_losses = self.model.train_state.loss_train
            loss_str = ",".join(map(str, current_losses))
            with open(self.file_path, "a") as f:
                f.write(f"{step},{loss_str}\n")


# ---------------------------------------------------------------------------------------------------------------
# Defining PINN model
# ---------------------------------------------------------------------------------------------------------------

layer_size = [3] + [128] * 4 + [4]   # 3 inputs (x (x1), y (x2), t), 4 outputs (u1, u2, eta1, eta2)
activation = "tanh" # Activation function
initializer = "Glorot uniform" # Initializer for weights and biases
net = dde.nn.FNN(layer_size, activation, initializer) #  Neural network
# Applying the scaling transformation
net.apply_output_transform(transform_func)

# Data object for the PDE
data = dde.data.TimePDE(
            geomtime, # Spatial and time domain
            pde, # Partial differential equation
            [bc_eta1_x, bc_flux_eta1_x, bc_eta2_x, bc_flux_eta2_x, bc_eta1_y, bc_flux_eta1_y, bc_eta2_y, bc_flux_eta2_y, ic_eta1, ic_eta2, observ_eta1, observ_eta2], # Boundary and initial conditions
            num_domain=20000, # Number of domain points
            num_boundary=4000, # Number of boundary points
            train_distribution='Hammersley', # Sequence for sampling points
            anchors=anchor_points, # Anchor points
            num_test=50000, # Number of testing points
        )

# model building
model = dde.Model(data, net)

# ---------------------------------------------------------------------------------------------------------------
# Defining configuration, and Training the PINN model
# ---------------------------------------------------------------------------------------------------------------

# Beta tracking with iterations
Beta_trainable.data = torch.tensor([1.0], dtype=torch.float32) # Initial beta value
Beta_tracker = dde.callbacks.VariableValue(Beta_trainable, period=100) # Tracking beta value
iterations_list = [0] # List to store iterations
beta_trainable_values = [Beta_trainable.item()] # List to store beta values
total_iterations = 0 # Total iterations

# File paths
base_dir = "FilePath" # Base directory for saving data
loss_path = os.path.join(base_dir, f"losses.txt") # Path for the loss file
beta_path = os.path.join(base_dir, f"beta_vs_iterations.txt") # Path for the beta file
eta_out_path = os.path.join(base_dir, f"eta_prediction.txt") # Path for the eta output
detailed_loss_tracker = SimpleLossTrackingCallback(file_path=loss_path)

# Training 
initial_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1000, 1000, 1000, 1000] # Weighting coefficients for the loss function
max_iterations = 30000 # Maximum number of iterations
while total_iterations < max_iterations:

    # Updating beta values every 100 iterations, different learning rates for precise estimation
    iter_this = 100
    if total_iterations < 10000:
        model.compile("adam", lr=1e-3, loss='MSE', loss_weights=initial_weights, external_trainable_variables=[Beta_trainable])
        losshistory, train_state = model.train(iterations=iter_this, batch_size=32, display_every=100, callbacks=[Beta_tracker, detailed_loss_tracker])

    elif total_iterations < 20000:
        model.compile("adam", lr=1e-4, loss='MSE', loss_weights=initial_weights, external_trainable_variables=[Beta_trainable])
        losshistory, train_state = model.train(iterations=iter_this, batch_size=32, display_every=1000, callbacks=[Beta_tracker, detailed_loss_tracker])
    else:  
        model.compile("adam", lr=1e-5, loss= 'MSE', loss_weights=initial_weights, external_trainable_variables=[Beta_trainable])
        losshistory, train_state = model.train(iterations=iter_this, batch_size=32, display_every=1000, callbacks=[Beta_tracker, detailed_loss_tracker])


    # Tracking beta and iteration values
    beta_trainable_values.append(Beta_trainable.item())
    total_iterations += iter_this
    iterations_list.append(total_iterations)

    # Intermediate beta values
    np.savetxt(beta_path, np.column_stack((iterations_list, beta_trainable_values)),
        fmt="%-12d %.10e", header="Iteration   Beta", comments='')
    
    print(f"Completed")


# ---------------------------------------------------------------------------------------------------------------
# Plotting the trainable beta values
# ---------------------------------------------------------------------------------------------------------------
                    
plt.figure(figsize=(10, 6))
plt.plot(iterations_list, beta_trainable_values, '-o', label='Beta', color='orange')
plt.yscale('log')  # Log scale
plt.xlabel('Iterations', fontsize=14)
plt.ylabel('Trainable Parameter Value (log scale)', fontsize=14)
plt.title('Trainable Parameters vs. Iterations', fontsize=16)
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()