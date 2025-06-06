# ---------------------------------------------------------------------------------------------------------------
# Importing Libraries and Configuration Setting
# ---------------------------------------------------------------------------------------------------------------
import os
os.environ["DDE_BACKEND"] = "pytorch"
import torch
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from deepxde.callbacks import Callback
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

# Gradient energy coefficient
Beta = 1e-4
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
u_scale = 1e-6 # Scaling factor for u1 and u2
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
    c_eta1_x = -(L * Beta) * eta1_x 
    c_eta1_y = -(L * Beta) * eta1_y 
    c_eta2_x = -(L * Beta) * eta2_x 
    c_eta2_y = -(L * Beta) * eta2_y 

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
bc_flux_eta2_x = PeriodicBC(geomtime, component_x=0, on_boundary=boundary_x, derivative_order=1, component=3)  # Gradient periodicity

# Periodic BCs in y (x2) direction
bc_eta1_y = PeriodicBC(geomtime, component_x=1, on_boundary=boundary_y, derivative_order=0, component=2)  # Function periodicity
bc_flux_eta1_y = PeriodicBC(geomtime, component_x=1, on_boundary=boundary_y, derivative_order=1, component=2)  # Gradient periodicity

bc_eta2_y = PeriodicBC(geomtime, component_x=1, on_boundary=boundary_y, derivative_order=0, component=3)  # Function periodicity
bc_flux_eta2_y = PeriodicBC(geomtime, component_x=1, on_boundary=boundary_y, derivative_order=1, component=3)  # Gradient periodicity


# ---------------------------------------------------------------------------------------------------------------
# Initial Conditions
# ---------------------------------------------------------------------------------------------------------------

# Initial condition function for eta1 for time step [0, T1]
def eta1_ic_func(X):
    x = X[:, 0:1] # x-coordinate (x1)
    y = X[:, 1:2] # y-coordinate (x2)
    return 0.1 + 0.4 * np.exp(-((x - 0.3)**2 + (y - 0.3)**2) / (2 * 0.05**2))

# Initial condition function for eta2 for time step [0, T1]
def eta2_ic_func(X):
    x = X[:, 0:1]
    y = X[:, 1:2]
    return 0.5 - 0.4 * np.exp(-((x - 0.3)**2 + (y - 0.3)**2) / (2 * 0.05**2))

# Initial condition applied at t = T_start
def initial_time(X, on_initial):
    return on_initial and np.isclose(X[2], 0.0)

# Initial conditions for eta1 and eta2
ic_eta1 = dde.IC(geomtime, eta1_ic_func, initial_time, component=2)
ic_eta2 = dde.IC(geomtime, eta2_ic_func, initial_time, component=3)


# Initial condition text data for time step [T1, Tn], generated from the previous time step as a text file.
ic_data = np.loadtxt(f"initial_condition_path", delimiter=',')

# Extracting spatial coordinates (x, y) from the text file
xy_ic = ic_data[:, 0:2]  # x, y coordinates
eta1_ic = ic_data[:, 2:3] # eta1 values
eta2_ic = ic_data[:, 3:4] # eta2 values
xyt_ic = np.hstack((xy_ic, np.zeros((xy_ic.shape[0], 1))))

# ic_eta1 = PointSetBC(xyt_ic, eta1_ic, component=2) # # Initial condition for time step [T1, Tn]
# ic_eta2 = PointSetBC(xyt_ic, eta2_ic, component=3)


# ---------------------------------------------------------------------------------------------------------------
# Adaptive Refinement Strategy
# ---------------------------------------------------------------------------------------------------------------
def adaptive_sampling_eta(model, geomtime, pde, num_samples, neighborhood_size):

    # Generating a large set of random points
    x = geomtime.random_points(100000)
    # Computing the residuals at these points
    residuals = model.predict(x, operator=pde)

    # Residuals for each ouput
    u1_residual = np.abs(residuals[0])  # First residual is for u1
    u2_residual = np.abs(residuals[1])  # Second residual is for u2
    eta1_residual = np.abs(residuals[2])  # Third residual is for eta1
    eta2_residual = np.abs(residuals[3])  # Fourth residual is for eta2

    # Combining the residuals
    residual_combined = u1_residual+ u2_residual+ eta1_residual + eta2_residual

    # Sorting the indices of the points by the highest combined residuals
    sorted_indices = np.argsort(-residual_combined.flatten())[:num_samples]

    # Selecting the top points with the highest errors 
    high_error_points = x[sorted_indices]

    # Ensuring unique sampling if required
    if neighborhood_size > 0:
        new_points = []
        for point in high_error_points:
            # Generating points around the high-error points within a defined neighborhood
            for _ in range(3):  # Generating multiple new points for each high-error point
                perturbation = np.random.uniform(-neighborhood_size, neighborhood_size, size=point.shape)
                new_point = point + perturbation
                new_point[:2] = np.clip(new_point[:2], geomtime.geometry.bbox[0], geomtime.geometry.bbox[1])
                new_points.append(new_point)
        
        new_points = np.array(new_points)
        # Combining high-error points with neighborhood-generated points
        high_error_points = np.vstack([high_error_points, new_points])

    # Ensuring uniqueness of the points before returning
    high_error_points = np.unique(high_error_points, axis=0)

    print(f"Selected {len(high_error_points)} unique high-error points")

    # Returning the selected points
    return high_error_points


# ---------------------------------------------------------------------------------------------------------------
# Defining Neural Network and Model Creation
# ---------------------------------------------------------------------------------------------------------------

layer_size = [3] + [128] * 4 + [4]  # 3 inputs (x (x1), y (x2), t), 4 outputs (u1, u2, eta1, eta2)
activation = "tanh" # Activation function
initializer = "Glorot uniform" # Initializer for weights and biases
net = dde.nn.FNN(layer_size, activation, initializer) #  Neural network
# Applying the scaling transformation
net.apply_output_transform(transform_func)

# Data object for the PDE
data = dde.data.TimePDE(
            geomtime, # Spatial and time domain
            pde, # Partial differential equation
            [bc_eta1_x, bc_flux_eta1_x, bc_eta2_x, bc_flux_eta2_x, bc_eta1_y, bc_flux_eta1_y, bc_eta2_y, bc_flux_eta2_y, ic_eta1, ic_eta2], # Boundary and initial conditions
            num_domain=30000, # Number of domain points
            num_boundary=4000, # Number of boundary points
            num_initial=4000, # Number of initial condition points
            train_distribution='Hammersley', # Distribution sequence for sampling points
            num_test=50000, # Number of testing points
        )

# Model building
model = dde.Model(data, net)


# ---------------------------------------------------------------------------------------------------------------
# Setting Up the Optimizer, Callbacks, and Loss Function, and model for training
# ---------------------------------------------------------------------------------------------------------------

# PDE resampler
pde_resampler = dde.callbacks.PDEPointResampler(period=2000, pde_points=True, bc_points=False)

# Weighting coefficients for the loss function
initial_weights =   [1, 1, 100, 100, 1, 1, 1, 1, 1, 1, 1, 1, 100, 100] 

# Settings for the lbfgs optimizer
dde.optimizers.config.set_LBFGS_options(
    maxcor=100,
    ftol=0,
    gtol=1e-12,
    maxiter=50000, # Maximum number of iterations
    maxfun=None,
    maxls=50,
)

# Model restoration for transfer learning
# model.compile("L-BFGS", loss='MSE', loss_weights=initial_weights)
# #model.restore("saved_model_path")

# Adam optimizer training 
model.compile("adam", lr=1e-3, loss='MSE', loss_weights=initial_weights)
losshistory, train_state = model.train(iterations=50000,batch_size=32, display_every=1000, callbacks=[pde_resampler], model_save_path=f"./model_path")

# L-BFGS optimizer training
model.compile("L-BFGS", loss='MSE', loss_weights=initial_weights)
losshistory, train_state = model.train(callbacks=[pde_resampler], model_save_path=f"./model_path")


# ---------------------------------------------------------------------------------------------------------------
# Training with adaptive refinement IF needed 
# ---------------------------------------------------------------------------------------------------------------

# Adaptive refinement parameters
num_sampling_rounds = 2  # number of sampling rounds
num_samples = 5000  # number of points to be added during each sampling round
for sampling_round in range(num_sampling_rounds):
    print(f"Sampling round {sampling_round + 1}/{num_sampling_rounds}")

    # Generating new high-error points
    x_new = adaptive_sampling_eta(model, geomtime, pde, num_samples=num_samples, neighborhood_size=0.01)

    # Converting existing training data points to a set for efficient lookup
    existing_points_set = set(map(tuple, model.data.train_x))
    print(f"Number of existing points: {len(existing_points_set)}")

    # Filtering out points that already exist in the training data
    unique_new_points = np.array([point for point in x_new if tuple(point) not in existing_points_set])

    # If there are any unique points, evaluating their errors
    if len(unique_new_points) > 0:
        # Predicting the residuals or errors at these points
        residuals = model.predict(unique_new_points, operator=pde)
        residuals_combined = np.sum([np.abs(res) for res in residuals], axis=0)

        # Sorting points by error magnitude (descending order)
        sorted_indices = np.argsort(-residuals_combined.flatten())
        top_error_indices = sorted_indices[:num_samples]

        # Selecting the top error points
        unique_new_points = unique_new_points[top_error_indices]

    # Adding the unique high-error points to the training data if any
    if len(unique_new_points) > 0:
        print(f"Adding {len(unique_new_points)} new unique points based on high-error regions")
        model.data.add_anchors(unique_new_points)

        # ADAM optimizer training with refinement strategy
        model.compile("adam", lr=1e-3, loss='MSE', loss_weights=initial_weights)
        losshistory, train_state = model.train(iterations=20000,batch_size=32, display_every=1000, callbacks=[pde_resampler], model_save_path=f"./time_models/model_time_step_adam_best")

        # L-BFGS optimizer training with refinement strategy
        model.compile("L-BFGS", loss='MSE', loss_weights=initial_weights)
        losshistory, train_state = model.train(callbacks=[pde_resampler], model_save_path=f"./time_models/model_time_step_final_best")
    else:
        print("No new unique points to add in this sampling round.")



# ---------------------------------------------------------------------------------------------------------------
# Loading reference data for comparison with PINN predictions for eta1 and eta2 at t = T_start, and t = T_end
# ---------------------------------------------------------------------------------------------------------------

# Reference text file for the initial time (t = T_start)
ref_data_start_time = np.loadtxt(f"Ref_t0_.txt", delimiter=',')

# Reference text file for the end time (t = T_end)
ref_data_end_time =np.loadtxt(f"Ref_t1_.txt", delimiter=',')

# Unpacking the data for the initial time, Note: data has four columns: x, y, eta1, eta2
start_time_data = (ref_data_start_time[:, 0], ref_data_start_time[:, 1], ref_data_start_time[:, 2], ref_data_start_time[:, 3])

# Unpacking the data for the end time, Note: data has four columns: x, y, eta1, eta2
end_time_data = (ref_data_end_time[:, 0], ref_data_end_time[:, 1], ref_data_end_time[:, 2], ref_data_end_time[:, 3])


# ---------------------------------------------------------------------------------------------------------------
# Plotting the comparison between Reference and PINN results for eta1 and eta2
# ---------------------------------------------------------------------------------------------------------------
def plot_comparison(start_time_data, end_time_data, model):

    # Defining min and max values based on Ref data
    T_start_domain = T_start  # Time step from Reference data
    T_end_domain = T_end  # Next time step in Reference data

    # Defining the plotting parameters
    Font_size = 9
    Axis_size = 8
    levels_number = 200
    cmap_color = 'jet'

    # Ref data 
    x_ref_start, y_ref_start, ref_eta1_start, ref_eta2_start = start_time_data
    x_ref_end, y_ref_end, ref_eta1_end, ref_eta2_end = end_time_data

    # Preparing input data for PINN predictions
    t_norm_start = np.full_like(x_ref_start, T_start_domain)
    X_input_start = np.column_stack((x_ref_start, y_ref_start, t_norm_start))

    t_norm_end = np.full_like(x_ref_end, T_end_domain)
    X_input_end = np.column_stack((x_ref_end, y_ref_end, t_norm_end))

    # Model prediction for both time steps
    predictions_start = model.predict(X_input_start)
    predictions_end = model.predict(X_input_end)

    # Extracting eta1 and eta2 predictions
    predicted_eta1_start = predictions_start[:, 2]
    predicted_eta2_start = predictions_start[:, 3]
    predicted_eta1_end = predictions_end[:, 2]
    predicted_eta2_end = predictions_end[:, 3]

    # Grid for contour plots
    grid_x_start = np.unique(x_ref_start)
    grid_y_start = np.unique(y_ref_start)
    X_start, Y_start = np.meshgrid(grid_x_start, grid_y_start)

    grid_x_end = np.unique(x_ref_end)
    grid_y_end = np.unique(y_ref_end)
    X_end, Y_end = np.meshgrid(grid_x_end, grid_y_end)

    # Ref and predicted data to fit the grid
    Reshaped_ref_eta1_start = ref_eta1_start.reshape(len(grid_y_start), len(grid_x_start))
    Reshaped_ref_eta2_start = ref_eta2_start.reshape(len(grid_y_start), len(grid_x_start))
    Reshaped_predicted_eta1_start = predicted_eta1_start.reshape(len(grid_y_start), len(grid_x_start))
    Reshaped_predicted_eta2_start = predicted_eta2_start.reshape(len(grid_y_start), len(grid_x_start))

    Reshaped_ref_eta1_end = ref_eta1_end.reshape(len(grid_y_end), len(grid_x_end))
    Reshaped_ref_eta2_end = ref_eta2_end.reshape(len(grid_y_end), len(grid_x_end))
    Reshaped_predicted_eta1_end = predicted_eta1_end.reshape(len(grid_y_end), len(grid_x_end))
    Reshaped_predicted_eta2_end = predicted_eta2_end.reshape(len(grid_y_end), len(grid_x_end))

    # Absolute error
    error_eta1_start = np.abs(Reshaped_predicted_eta1_start - Reshaped_ref_eta1_start)
    error_eta2_start = np.abs(Reshaped_predicted_eta2_start - Reshaped_ref_eta2_start)
    error_eta1_end = np.abs(Reshaped_predicted_eta1_end - Reshaped_ref_eta1_end)
    error_eta2_end = np.abs(Reshaped_predicted_eta2_end - Reshaped_ref_eta2_end)

    # Creating plots
    plt.figure(figsize=(10, 14))

    def plot_subplot(index, X, Y, data, title):
        plt.subplot(4, 3, index)
        contour = plt.contourf(X, Y, data, levels=levels_number, cmap=cmap_color)
        cbar = plt.colorbar(contour)
        cbar.ax.tick_params(labelsize=Axis_size)
        # plt.title(title, fontsize=Font_size, fontweight='bold')
        # plt.xticks(fontsize=Axis_size)
        # plt.yticks(fontsize=Axis_size)
        plt.axis("off")

    # Ref results (eta1 and eta2)
    plot_subplot(1, X_start, Y_start, Reshaped_ref_eta1_start, f'Ref $\\eta_1$ at $t$={T_start_domain}')
    plot_subplot(4, X_end, Y_end, Reshaped_ref_eta1_end, f'Ref $\\eta_1$ at $t$={T_end_domain}')
    plot_subplot(7, X_start, Y_start, Reshaped_ref_eta2_start, f'Ref $\\eta_2$ at $t$={T_start_domain}')
    plot_subplot(10, X_end, Y_end, Reshaped_ref_eta2_end, f'Ref $\\eta_2$ at $t$={T_end_domain}')

    # PINN results (eta1 and eta2)
    plot_subplot(2, X_start, Y_start, Reshaped_predicted_eta1_start, f'PINN $\\eta_1$ at $t$={T_start_domain}')
    plot_subplot(5, X_end, Y_end, Reshaped_predicted_eta1_end, f'PINN $\\eta_1$ at $t$={T_end_domain}')
    plot_subplot(8, X_start, Y_start, Reshaped_predicted_eta2_start, f'PINN $\\eta_2$ at $t$={T_start_domain}')
    plot_subplot(11, X_end, Y_end, Reshaped_predicted_eta2_end, f'PINN $\\eta_2$ at $t$={T_end_domain}')

    # Absolute error (eta1 and eta2)
    plot_subplot(3, X_start, Y_start, error_eta1_start, f'Error $|\\eta_1|$ at $t$={T_start_domain}')
    plot_subplot(6, X_end, Y_end, error_eta1_end, f'Error $|\\eta_1|$ at $t$={T_end_domain}')
    plot_subplot(9, X_start, Y_start, error_eta2_start, f'Error $|\\eta_2|$ at $t$={T_start_domain}')
    plot_subplot(12, X_end, Y_end, error_eta2_end, f'Error $|\\eta_2|$ at $t$={T_end_domain}')

    # Final adjustments
    plt.tight_layout()
    plt.show()

# Calling function to compare Ref and PINN results
plot_comparison(start_time_data, end_time_data, model)


# ---------------------------------------------------------------------------------------------------------------
# Plotting All Four Outputs (u1, u2, eta1, eta2)
# ---------------------------------------------------------------------------------------------------------------

# Function to plot the results for u1, u2, eta1, and eta2 
def plot_all_four_outputs(model, t_start, t_end, grid=101):

    # Define the parameters for the plot
    T_start_domain = 0
    T_end_domain = 1.0
    Font_size = 9
    pade_szie = 0
    Axis_size = 8
    levels_number = 100

    # Create a grid of points over x and y
    x = np.linspace(0, 1, grid)
    y = np.linspace(0, 1, grid)
    X, Y = np.meshgrid(x, y)

    # Flatten the grid to pass into the model
    X_flat = X.flatten()[:, None]
    Y_flat = Y.flatten()[:, None]

    # Prepare time arrays for t_start and t_end
    t_start_array = np.full_like(X_flat, t_start)
    t_end_array = np.full_like(X_flat, t_end)

    # Preparing input data for both time instances
    input_start = np.hstack((X_flat, Y_flat, t_start_array))
    input_end = np.hstack((X_flat, Y_flat, t_end_array))

    # Prediction using the trained model for both start and end times
    pred_start = model.predict(input_start)
    pred_end = model.predict(input_end)

    # Reshape predictions into grid format for the 4 outputs (u1, u2, eta1, eta2)
    u1_start, u2_start, eta1_start, eta2_start = np.split(pred_start, 4, axis=1)  # t_start
    u1_end, u2_end, eta1_end, eta2_end = np.split(pred_end, 4, axis=1)  # t_end

    # Reshaping for plotting
    u1_start = u1_start.reshape(X.shape)
    u2_start = u2_start.reshape(X.shape)
    eta1_start = eta1_start.reshape(X.shape)
    eta2_start = eta2_start.reshape(X.shape)

    u1_end = u1_end.reshape(X.shape)
    u2_end = u2_end.reshape(X.shape)
    eta1_end = eta1_end.reshape(X.shape)
    eta2_end = eta2_end.reshape(X.shape)

    # Creating the figure with a 4x2 grid layout
    plt.figure(figsize=(14, 5))

    # Plotting u1 at the start time
    plt.subplot(2, 4, 1)
    contour1 = plt.contourf(X, Y, u1_start, levels=100, cmap='jet')
    cbar1 = plt.colorbar(contour1)
    cbar1.ax.tick_params(labelsize=Axis_size)
    plt.title(fr'$\mathbf{{{{u_1}}}}$ at $\mathbf{{{{t}}}}$ = {T_start_domain}', fontsize=Font_size, fontweight='bold', pad=pade_szie)
    plt.xticks(fontsize=Axis_size)
    plt.yticks(fontsize=Axis_size)

    # Plotting u1 at the end time
    plt.subplot(2, 4, 5)
    contour2 = plt.contourf(X, Y, u1_end, levels=100, cmap='jet')
    cbar2 = plt.colorbar(contour2)
    cbar2.ax.tick_params(labelsize=Axis_size)
    plt.title(fr'$\mathbf{{{{u_1}}}}$ at $\mathbf{{{{t}}}}$ = {T_end_domain}', fontsize=Font_size, fontweight='bold', pad=pade_szie)
    plt.xticks(fontsize=Axis_size)
    plt.yticks(fontsize=Axis_size)

    # Plotting u2 at the start time
    plt.subplot(2, 4, 2)
    contour3 = plt.contourf(X, Y, u2_start, levels=100, cmap='jet')
    cbar3 = plt.colorbar(contour3)
    cbar3.ax.tick_params(labelsize=Axis_size)
    plt.title(fr'$\mathbf{{{{u_2}}}}$ at $\mathbf{{{{t}}}}$ = {T_start_domain}', fontsize=Font_size, fontweight='bold', pad=pade_szie)
    plt.xticks(fontsize=Axis_size)
    plt.yticks(fontsize=Axis_size)

    # Plotting u2 at the end time
    plt.subplot(2, 4, 6)
    contour4 = plt.contourf(X, Y, u2_end, levels=100, cmap='jet')
    cbar4 = plt.colorbar(contour4)
    cbar4.ax.tick_params(labelsize=Axis_size)
    plt.title(fr'$\mathbf{{{{u_2}}}}$ at $\mathbf{{{{t}}}}$ = {T_end_domain}', fontsize=Font_size, fontweight='bold', pad=pade_szie)
    plt.xticks(fontsize=Axis_size)
    plt.yticks(fontsize=Axis_size)

    # Plotting eta1 at the start time
    plt.subplot(2, 4, 3)
    contour5 = plt.contourf(X, Y, eta1_start, levels=100, cmap='jet')
    cbar5 = plt.colorbar(contour5)
    cbar5.ax.tick_params(labelsize=Axis_size)  
    plt.title(fr'$\mathbf{{{{\eta_1}}}}$ at $\mathbf{{{{t}}}}$ = {T_start_domain}', fontsize=Font_size, fontweight='bold', pad=pade_szie)
    plt.xticks(fontsize=Axis_size)
    plt.yticks(fontsize=Axis_size)

    # Plotting eta1 at the end time
    plt.subplot(2, 4, 7)
    contour6 = plt.contourf(X, Y, eta1_end, levels=100, cmap='jet')
    cbar6 = plt.colorbar(contour6)
    cbar6.ax.tick_params(labelsize=Axis_size)  
    plt.title(fr'$\mathbf{{{{\eta_1}}}}$ at $\mathbf{{{{t}}}}$ = {T_end_domain}', fontsize=Font_size, fontweight='bold', pad=pade_szie)
    plt.xticks(fontsize=Axis_size)
    plt.yticks(fontsize=Axis_size)


    # Plotting eta2 at the start time
    plt.subplot(2, 4, 4)
    contour7 = plt.contourf(X, Y, eta2_start, levels=100, cmap='jet')
    cbar7 = plt.colorbar(contour7)
    cbar7.ax.tick_params(labelsize=Axis_size)  
    plt.title(fr'$\mathbf{{{{\eta_2}}}}$ at $\mathbf{{{{t}}}}$ = {T_start_domain}', fontsize=Font_size, fontweight='bold', pad=pade_szie)
    plt.xticks(fontsize=Axis_size)
    plt.yticks(fontsize=Axis_size)

    # Plotting eta2 at the end time
    plt.subplot(2, 4, 8)
    contour8 = plt.contourf(X, Y, eta2_end, levels=100, cmap='jet')
    cbar8 = plt.colorbar(contour8)
    cbar8.ax.tick_params(labelsize=Axis_size)  
    plt.title(fr'$\mathbf{{{{\eta_2}}}}$ at $\mathbf{{{{t}}}}$ = {T_end_domain}', fontsize=Font_size, fontweight='bold', pad=pade_szie)
    plt.xticks(fontsize=Axis_size)
    plt.yticks(fontsize=Axis_size)

    # Adjustting  layout
    plt.tight_layout()
    plt.show()

# Calling the function
plot_all_four_outputs(model, T_start, T_end, grid=101)

# ----------------------------------------------------------------------------------------------
# Function to create the initial condition text file for eta1 and eta2 for the next time step
# ----------------------------------------------------------------------------------------------
def save_etaIC_from_model(model, grid_size, filename):

    # Creating the grid over the [0, 1] x [0, 1] domain
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x, y)

    # Flatten the grid
    X_flat = X.flatten()[:, None]
    Y_flat = Y.flatten()[:, None]
    
    # Setting time input to t = 1 for the next time domain
    t = np.ones_like(X_flat) 
    
    # Stacking x, y, t
    inputs = np.hstack((X_flat, Y_flat, t))

    # Predicting eta1 and eta2 
    predictions = model.predict(inputs)
    
    # Splitting the predictions for eta1 and eta2 (3rd and 4th columns of the output)
    eta1, eta2 = predictions[:, 2], predictions[:, 3]

    # File content in the format (x, y, eta1, eta2)
    output_lines = []
    for i in range(len(X_flat)):
        line = f"{X_flat[i, 0]:.6f},{Y_flat[i, 0]:.6f},{eta1[i]:.5f},{eta2[i]:.5f}\n"
        output_lines.append(line)
    
    # Saving file
    with open(filename, "w") as f:
        f.writelines(output_lines)
    
    print(f"Data saved to {filename}")

# Function to save the data
grid_size = 101
output_filename = "output_text_file_directory"
save_etaIC_from_model(model, grid_size, output_filename)

# ----------------------------------------------------------------------------------------------
# End
# ----------------------------------------------------------------------------------------------