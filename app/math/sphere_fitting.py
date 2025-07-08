import numpy as np
from scipy.optimize import least_squares

def fit_sphere(points):
    """
    Fit a sphere to 3D points using least squares optimization.
    
    Parameters:
    points (np.ndarray): Nx3 array of 3D points
    
    Returns:
    tuple: (center_x, center_y, center_z, radius)
    """
    def residuals(params, points):
        """
        Calculate residuals (distance from points to sphere surface)
        """
        x0, y0, z0, r = params
        return np.sqrt((points[:,0] - x0)**2 + (points[:,1] - y0)**2 + (points[:,2] - z0)**2) - r
    
    # Initial guess: center at mean of points, radius as average distance to center
    x0 = np.mean(points[:,0])
    y0 = np.mean(points[:,1])
    z0 = np.mean(points[:,2])
    r0 = np.mean(np.sqrt((points[:,0]-x0)**2 + (points[:,1]-y0)**2 + (points[:,2]-z0)**2))
    
    # Perform least squares optimization
    result = least_squares(residuals, [x0, y0, z0, r0], args=(points,))
    
    return result.x

if __name__ == "__main__":    

    # Example usage:
    # Generate some points on a sphere with some noise
    np.random.seed(42)
    true_center = [2, 3, 4]
    true_radius = 5
    num_points = 100

    # Generate spherical coordinates
    theta = np.random.uniform(0, 2*np.pi, num_points)
    phi = np.random.uniform(0, np.pi, num_points)

    # Convert to cartesian coordinates with noise
    x = true_center[0] + true_radius * np.sin(phi) * np.cos(theta) + np.random.normal(0, 0.1, num_points)
    y = true_center[1] + true_radius * np.sin(phi) * np.sin(theta) + np.random.normal(0, 0.1, num_points)
    z = true_center[2] + true_radius * np.cos(phi) + np.random.normal(0, 0.1, num_points)

    points = np.column_stack((x, y, z))

    # Fit sphere
    center_x, center_y, center_z, radius = fit_sphere(points)

    print(f"Fitted center: ({center_x:.3f}, {center_y:.3f}, {center_z:.3f})")
    print(f"Fitted radius: {radius:.3f}")
    print(f"True center: {true_center}")
    print(f"True radius: {true_radius}")