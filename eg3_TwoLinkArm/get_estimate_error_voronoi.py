import os
import sys
import numpy as np
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import matplotlib
import time
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull, HalfspaceIntersection
from scipy.optimize import linprog
import pickle
import torch

from cores.lip_nn.models import NeuralNetwork
from cores.dynamical_systems.create_system import get_system
from cores.utils.utils import get_nn_config, load_nn_weights, format_time
from cores.utils.config import Configuration
from cores.dataloader.dataset_utils import DynDataset

def find_point_inside_region(A, b):
    """
    Find a point clearly inside the region defined by the halfspaces A * x + b <= 0 using linear programming.

    Parameters:
    A : np.ndarray
        The matrix of normal vectors for the inequalities.
    b : np.ndarray
        The vector of bias terms for the inequalities.

    Returns:
    res.x[:-1] : np.ndarray
        The point x that is clearly inside the region.
    """
    # Number of variables (dimension of space)
    num_vars = A.shape[1]
    
    # Augment A and b to include margin t
    # We need to add an extra column to A for the variable t, which is to be maximized
    c = np.zeros(num_vars + 1)  # Objective function: Maximize t
    c[-1] = -1  # Maximize the margin t (minimizing -t is the same)

    # Modify A to include t in the constraints A * x + b <= -t
    A_lp = np.hstack([A, np.ones((A.shape[0], 1))])  # Add a column of 1's for t
    b_lp = -b  # Shift the inequalities to account for t

    # Bounds for the variables (x and t)
    bounds = [(None, None)] * num_vars + [(0, None)]  # t >= 0

    # Solve the linear program
    res = linprog(c, A_ub=A_lp, b_ub=b_lp, bounds=bounds, method='highs')

    if res.success:
        # Return the solution for x (excluding t, which is the last component)
        return res.x[:-1], res.x[-1]  # The point x and the margin t
    else:
        raise ValueError("Linear programming failed to find a solution")

def get_surrounding_ridge_equations(vor, point_index):
    """
    Compute the stacked inequalities of all surrounding ridges (hyperplanes) for a given point in an N-dimensional Voronoi diagram.
    Returns the inequalities of Ax + b <= 0 in the form [A; b], where A is the matrix of normal vectors and b is the bias vector.

    Parameters:
    vor : scipy.spatial.Voronoi
        The Voronoi diagram object computed using `scipy.spatial.Voronoi`.
    point_index : int
        Index of the point for which to compute the inequalities.

    Returns:
    A stacked matrix of the form [A; b], where each row is the normal vector followed by the bias term.
    """
    inequalities = []
    
    # Get the neighbors of the given point
    neighbors = np.where(vor.ridge_points == point_index)[0]
    neighbor_indices = vor.ridge_points[neighbors].flatten()
    neighbor_indices = neighbor_indices[neighbor_indices != point_index]  # Filter out the selected point

    # Iterate over each neighbor to compute the normalized ridge inequalities
    for neighbor_index in neighbor_indices:
        P1 = vor.points[point_index]   # The given point
        P2 = vor.points[neighbor_index]  # The neighboring point

        # Midpoint
        M = (P1 + P2) / 2

        # Normal vector (difference between the two points)
        n = P2 - P1

        # Compute the right-hand side of the equation (n . M)
        rhs = np.dot(n, M)

        # Normalize the normal vector
        norm = np.linalg.norm(n)
        n_normalized = n / norm  # Normalize the normal vector
        rhs_normalized = rhs / norm  # Adjust the right-hand side

        # Stacking A (normalized vector) with b (-rhs_normalized)
        inequality = np.append(n_normalized, -rhs_normalized)  # Append the bias term to the normal vector

        # Store the inequality
        inequalities.append(inequality)

    # Convert the list of inequalities into a numpy array
    return np.array(inequalities)

def get_vertices_inside_convex_hull(hull, ridge_equations):
    """
    Given a convex hull and a set of ridge equations, return the vertices inside the convex hull.
    
    Parameters:
        hull (ConvexHull): The convex hull object.
        ridge_equations (array-like): The stacked inequalities of the surrounding ridges.
    
    Returns:
        vertices (list of tuple): List of vertices inside the convex hull.
    """
    hull_equations = hull.equations
    equations = np.concatenate((hull_equations, ridge_equations), axis=0)
    feasible_point = find_point_inside_region(equations[:, :-1], equations[:, -1])[0]
    hsi = HalfspaceIntersection(equations, feasible_point)
    
    return hsi.intersections

def get_surrounding_vertices(vor, point_index):
    """
    Given a Voronoi diagram and the index of a point, return the surrounding vertices of the Voronoi cell,
    ignoring vertices at infinity.
    
    Parameters:
        vor (Voronoi): The Voronoi diagram object.
        point_index (int): The index of the point in the input points.
    
    Returns:
        vertices (list of tuple): List of the finite vertices surrounding the given point.
    """
    region_index = vor.point_region[point_index]
    region = vor.regions[region_index]
    
    # Ignore vertices at infinity (-1)
    finite_region = [i for i in region if i != -1]
    vertices = [vor.vertices[i] for i in finite_region]
    
    return vertices

# def get_surrounding_vertices(vor, point_index):
#     """
#     Given a Voronoi diagram and the index of a point, return the surrounding vertices of the Voronoi cell,
#     ignoring vertices at infinity.
    
#     Parameters:
#         vor (Voronoi): The Voronoi diagram object.
#         point_index (int): The index of the point in the input points.
    
#     Returns:
#         vertices (list of tuple): List of the finite vertices surrounding the given point.
#     """
#     region_index = vor.point_region[point_index]
#     region = vor.regions[region_index]
    
#     if -1 in region:
#         return []
#     else:
#         vertices = [vor.vertices[i] for i in region]
    
#     return vertices


def is_inside_convex_hull(hull, point):
    """
    Check if a point is inside the convex hull.
    
    Parameters:
        hull (ConvexHull): The convex hull object.
        point (array-like): The point to check.
    
    Returns:
        is_inside (bool): True if the point is inside the convex hull, False otherwise.
    """

    values = hull.equations[:, :-1] @ point + hull.equations[:, -1]
    return not np.any(values > 0)

def get_estimate_error(exp_num, dataset_num, dataset, unique_indices, vor, hull, system_lipschitz, selected_indices):
    print("==> Exp Num:", exp_num)
    results_dir = "{}/eg3_results/{:03d}".format(str(Path(__file__).parent.parent), exp_num)
    if not os.path.exists(results_dir):
        results_dir = "{}/eg3_results/{:03d}_keep".format(str(Path(__file__).parent.parent), exp_num)
    test_settings_path = os.path.join(results_dir, "test_settings_{:03d}.json".format(exp_num))
    with open(test_settings_path, "r", encoding="utf8") as f:
        test_settings = json.load(f)

    # Check if dataset matches the dataset_num
    if dataset_num != int(test_settings["train_config"]["dataset"]):
        print("==> Skipping... Dataset number does not match the test settings")
        return
    
    # Decide torch device
    config = Configuration()
    device = config.device
    print('==> torch device: ', device)
    
    # Build neural network
    nn_config = test_settings["nn_config"]
    input_bias = np.array(nn_config["input_bias"], dtype=config.np_dtype)
    input_transform = np.array(nn_config["input_transform_to_inverse"], dtype=config.np_dtype)
    input_transform = np.zeros_like(input_transform)
    output_transform = np.array(nn_config["output_transform"], dtype=config.np_dtype)
    train_transform = bool(nn_config["train_transform"])
    zero_at_zero = bool(nn_config["zero_at_zero"])

    nn_config = get_nn_config(nn_config)
    if nn_config.layer == 'Lip_Reg':
        model = NeuralNetwork(nn_config, input_bias=None, input_transform=None, output_transform=None, train_transform=train_transform, zero_at_zero=False)
    else:
        model = NeuralNetwork(nn_config, input_bias, input_transform, output_transform, train_transform, zero_at_zero)

    model = load_nn_weights(model, os.path.join(results_dir, 'nn_best.pt'), device)
    model.eval()

    # Build dynamical system
    system_name = test_settings["nominal_system_name"]
    nominal_system = get_system(system_name).to(device)

    x_np = dataset.x.cpu().numpy()[unique_indices]
    u_np = dataset.u.cpu().numpy()[unique_indices]
    x_dot_np = dataset.x_dot.cpu().numpy()[unique_indices]
    x_dot_np = x_dot_np[:,selected_indices]
    x_torch = torch.tensor(x_np, dtype=config.pt_dtype, device=device)
    u_torch = torch.tensor(u_np, dtype=config.pt_dtype, device=device)
    x_dot_pred_torch = model(x_torch) + nominal_system(x_torch, u_torch)[:,selected_indices]
    x_dot_pred_np = x_dot_pred_torch.cpu().detach().numpy()
    pred_error_np = np.linalg.norm(x_dot_pred_np - x_dot_np, 2, axis=1)

    # Collects the errors
    data = {}

    hull_A = hull.equations[:, :-1]
    hull_b = hull.equations[:, -1]

    # Global lipschitz error
    global_lipschitz_file  = "{}/global_lipschitz.pkl".format(results_dir)
    if os.path.exists(global_lipschitz_file):
        print("==> Global lipschitz file exists")
        with open(global_lipschitz_file, "rb") as f:
            global_lipschitz_data = pickle.load(f)
        global_lipschitz = global_lipschitz_data["global_lipschitz"]
        global_lipschitz_error = 0.0
        start_time = time.time()
        total_points = x_np.shape[0]
        for point_index in range(x_np.shape[0]):
            anchor_point = x_np[point_index]
            max_distance = 0.0
            vertices = get_surrounding_vertices(vor, point_index)
            for vertex in vertices:
                if not is_inside_convex_hull(hull, vertex):
                    continue
                distance = np.linalg.norm(vertex - anchor_point, 2)
                max_distance = max(max_distance, distance)
            error = pred_error_np[point_index] + (system_lipschitz + global_lipschitz) * max_distance
            global_lipschitz_error = max(global_lipschitz_error, error)
            if point_index % 100 == 0:
                print("Visited points: {:06d}/{:06d}. Time per point {:.4f}s. global_lipschitz_error: {:.3f}.".format(point_index+1, total_points, \
                    (time.time() - start_time) / (point_index + 1), global_lipschitz_error))
        total_time = time.time() - start_time
        print("Total time: {}".format(total_time))
        data["global_lipschitz_error"] = global_lipschitz_error
        data["global_lipschitz_time"] = total_time

    # LipSDP lipschitz error
    lipsdp_lipschitz_file  = "{}/lipsdp_lipschitz.pkl".format(results_dir)
    if os.path.exists(lipsdp_lipschitz_file):
        print("==> LipSDP lipschitz file exists")
        with open(lipsdp_lipschitz_file, "rb") as f:
            lipsdp_lipschitz_data = pickle.load(f)
        lipsdp_lipschitz = lipsdp_lipschitz_data["lipsdp_lipschitz"]
        lipsdp_lipschitz_error = 0.0
        start_time = time.time()
        total_points = x_np.shape[0]
        for point_index in range(x_np.shape[0]):
            anchor_point = x_np[point_index]
            max_distance = 0.0
            vertices = get_surrounding_vertices(vor, point_index)
            for vertex in vertices:
                if not is_inside_convex_hull(hull, vertex):
                    continue
                distance = np.linalg.norm(vertex - anchor_point, 2)
                max_distance = max(max_distance, distance)
            error = pred_error_np[point_index] + (system_lipschitz + lipsdp_lipschitz) * distance
            lipsdp_lipschitz_error = max(lipsdp_lipschitz_error, error)
            if point_index % 10000 == 0:
                print("Visited points: {:06d}/{:06d}. Time per point {:.4f}s. lipsdp_lipschitz_error: {:.3f}.".format(point_index+1, total_points, \
                    (time.time() - start_time) / (point_index + 1), lipsdp_lipschitz_error))
        total_time = time.time() - start_time
        print("Total time: {}".format(format_time(total_time)))
        data["lipsdp_lipschitz_error"] = lipsdp_lipschitz_error
        data["lipsdp_lipschitz_time"] = total_time
    
    with open("{}/estimate_error_voronoi.pkl".format(results_dir), "wb") as f:
        pickle.dump(data, f)
    print("==> Saved the estimate errors. Process completed.")
    
    
if __name__ == "__main__":
    dataset_num = 1
    dataset_folder = "{}/datasets/eg3_TwoLinkArm/{:03d}".format(str(Path(__file__).parent.parent), dataset_num)
    dataset_file = "{}/dataset.mat".format(dataset_folder)
    config = Configuration()
    dataset = DynDataset(dataset_file, config)

    x_np = dataset.x.cpu().numpy()
    _, unique_indices = np.unique(x_np, return_index=True, axis=0)
    x_np = x_np[unique_indices]
    print("==> Number of unique data points: ", x_np.shape[0])

    print("==> Computing Convex Hull...")
    hull = ConvexHull(x_np)

    if not os.path.exists("{}/voronoi.pkl".format(dataset_folder)):
        print("==> Computing Voronoi diagram...")
        time_start = time.time()
        vor = Voronoi(x_np)
        print("==> Time computing Voronoi diagram: {}".format(format_time(time.time() - time_start)))
        with open("{}/voronoi.pkl".format(dataset_folder), "wb") as f:
            pickle.dump(vor, f)
    else:
        print("==> Loading Voronoi diagram...")
        with open("{}/voronoi.pkl".format(dataset_folder), "rb") as f:
            vor = pickle.load(f)

    # !!!! Modify these values according to the system !!!! 
    system_lipschitz = 0.29
    selected_indices = [2, 3]

    exp_nums = [81, 161, 162, 163, 164]
    for exp_num in exp_nums:
        get_estimate_error(exp_num, dataset_num, dataset, unique_indices, vor, hull, system_lipschitz, selected_indices)
        print("#"*100)