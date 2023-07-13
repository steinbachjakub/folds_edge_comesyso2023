import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from scipy.spatial import ConvexHull
import cvxpy as cp
import cdd as pcdd
from matplotlib import pyplot as plt
from ultralytics import YOLO

def find_inscribed_ellipsiod(mask):
    """
    Function that finds a convex hull of given array of coordinates and then finds the maximum volume ellipsiod based on
    the convex hull.
    :param mask: Points returned by the model inference
    :return:
    """
    # Find a convex hull of the mask
    hull = ConvexHull(mask)
    pts = mask[hull.vertices]
    npoints = pts.shape[0]

    # Make the V-representation of the points; you have to prepend the points with a column of ones
    V = np.column_stack((np.ones(npoints), pts))
    mat = pcdd.Matrix(V, number_type='fraction') # use fractions if possible
    mat.rep_type = pcdd.RepType.GENERATOR
    poly = pcdd.Polyhedron(mat)

    # H-representation of the points: obtain matrix A and vector b such that the polyhedron is defined by Ax <= b
    H = poly.get_inequalities()
    # get A and b
    G = np.asarray(H[0:])
    b = G[:,0]
    A = -G[:,1:3]

    # now we define the convex optimization problem
    n = 2 # dimension
    # variables
    Bvar = cp.Variable((n, n), symmetric=True) # n x n symmetric matrix
    dvar = cp.Variable(n)      # vector
    # constraints
    constraints = [] # to store the constraints
    for i in range(npoints):
        constraints.append(
            cp.norm(Bvar @ A[i,:]) + A[i,:] @ dvar <= b[i]
        )
    # objective
    objective = cp.Minimize(-cp.log_det(Bvar))
    # problem
    problem = cp.Problem(objective, constraints)
    # solve problem
    x = problem.solve(solver=cp.SCS)
    # solutions
    variables = problem.solution.primal_vars

    B = Bvar.value
    d = dvar.value
    N = int(1e3)
    dn = np.transpose(np.repeat([d], N, axis=0))
    t = np.linspace(0, 2 * np.pi, N)
    u = np.array([np.cos(t), np.sin(t)])
    coords = B @ u + dn

    center_dists = []
    for point in coords.T:
        center_dists.append(np.linalg.norm(point - d))

    center_dists = np.array(center_dists)



    dic_results = {"a": max(center_dists),
                   "b": min(center_dists),
                   "hull": pts.astype(int),
                   "ellipse": coords.T.astype(int),
                   "center": d}
    return dic_results


if __name__ == "__main__":
    data_paths = [Path("data","labeled_data","test","images","0cc5905d-3_poop_stitna_zlaza_frame_776_0000_jpg.rf.e601487c938cfc5bb7b3dffeddfe0ba5.jpg"),
                  Path("data","labeled_data","test","images","2c342dac-2_pareza_frame_790_0000_jpg.rf.ebdd7d7d442c42ec468dc774638f48ea.jpg")]
    # Load a model
    model = YOLO("segmentation_model.pt")
    # Inference
    inference = model(data_paths, device="cpu")
    for img_path, result in zip(data_paths, inference):
        # Mask
        mask = result.masks.xy[0]
        dic_results = find_inscribed_ellipsiod(mask)

        # Drawing an image
        img = cv2.imread(str(img_path))
        img = cv2.polylines(img, [dic_results["hull"].astype(int)], True, (0, 0, 255), 1)

        for point in dic_results["ellipse"]:
            img = cv2.circle(img, (int(point[0]), int(point[1])), 1, (0, 255, 255), -1)

        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()