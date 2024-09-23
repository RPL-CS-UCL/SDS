import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

def compute_target_align_error(aligned_points, target_points):

    assert aligned_points.shape == target_points.shape, "Shape of aligned points and target points must match."

    distances = np.linalg.norm(aligned_points - target_points, axis=1)

    mean_distance = np.mean(distances)
    
    return distances, mean_distance

def icp_2d(source_points, target_points, max_iterations=100, tolerance=1e-5):
    final_transformation = np.eye(3)
    
    for _ in range(max_iterations):
        tree = KDTree(target_points)
        distances, indices = tree.query(source_points)
        closest_points = target_points[indices]

        centroid_source = np.mean(source_points, axis=0)
        centroid_target = np.mean(closest_points, axis=0)

        centered_source = source_points - centroid_source
        centered_target = closest_points - centroid_target

        H = centered_source.T @ centered_target

        U, S, Vt = np.linalg.svd(H)
        rotation_matrix = Vt.T @ U.T

        if np.linalg.det(rotation_matrix) < 0:
            Vt[1, :] *= -1
            rotation_matrix = Vt.T @ U.T

        translation_vector = centroid_target - rotation_matrix @ centroid_source

        source_points = (rotation_matrix @ source_points.T).T + translation_vector

        current_transformation = np.eye(3)
        current_transformation[:2, :2] = rotation_matrix
        current_transformation[:2, 2] = translation_vector

        final_transformation = current_transformation @ final_transformation

        if np.mean(distances) < tolerance:
            break

    return final_transformation, source_points


def plot_points(src,tgt,aligned,background=None,output_path='./icp_2d_plot.png'):
    if background is None:  
        plt.figure(figsize=(8, 8))
    else:
        # TODO Pass in background image
        pass
    plt.scatter(src[:, 0], src[:, 1], c='r', label='Source Points')
    plt.scatter(tgt[:, 0], tgt[:, 1], c='g', label='Target Points')
    plt.scatter(aligned[:, 0], aligned[:, 1], c='b', label='Aligned Source Points', marker='+')
    plt.legend()
    plt.title('ICP Algorithm: Source, Target, and Aligned Points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)

    plt.savefig(output_path)
    plt.show()


if __name__=="__main__":
    source_points = np.random.rand(100, 2)
    angle = np.radians(30)
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    translation = np.array([0.5, 0.3])
    target_points = (rotation_matrix @ source_points.T).T + translation
    transformation_matrix, aligned_source_points = icp_2d(source_points.copy(), target_points)
    
    plot_points(source_points,target_points,aligned_source_points)
