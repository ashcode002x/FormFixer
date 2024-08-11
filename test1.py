import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from scipy.special import comb
from scipy.optimize import curve_fit
from matplotlib.patches import Ellipse
from skimage.measure import EllipseModel

# Load CSV file


def load_csv(file_path):
    """Load data from CSV file."""
    df = pd.read_csv(file_path, header=None)
    return df.values

# RDP Algorithm for Curve Simplification


def rdp(points, epsilon):
    """Ramer-Douglas-Peucker algorithm for curve simplification."""
    def perpendicular_distance(pt, line_start, line_end):
        """Compute the perpendicular distance from a point to a line segment."""
        if np.all(line_start == line_end):
            return np.linalg.norm(pt - line_start)
        return np.abs(np.cross(line_end - line_start, pt - line_start)) / np.linalg.norm(line_end - line_start)

    if len(points) < 3:
        return points

    def rdp_rec(points, epsilon):
        dmax = 0
        index = 0
        for i in range(1, len(points) - 1):
            d = perpendicular_distance(points[i], points[0], points[-1])
            if d > dmax:
                index = i
                dmax = d

        if dmax >= epsilon:
            rec_results1 = rdp_rec(points[:index + 1], epsilon)
            rec_results2 = rdp_rec(points[index:], epsilon)
            return np.vstack((rec_results1[:-1], rec_results2))
        else:
            return np.vstack((points[0], points[-1]))

    return rdp_rec(points, epsilon)

# Check if points form a straight line


def is_potential_line(points, threshold=0.1):
    """Check if the points are potentially a straight line."""
    if len(points) < 2:
        return True
    X = points[:, 0].reshape(-1, 1)
    y = points[:, 1]
    model = RANSACRegressor()
    model.fit(X, y)
    line_error = np.mean(np.abs(y - model.predict(X)))
    return line_error < threshold

# Convert points to a perfect straight line


def make_straight_line(points):
    """Convert points to a perfect straight line between the start and end points."""
    start_point = points[0]
    end_point = points[-1]
    return np.vstack([start_point, end_point])

# Bezier Curve Fitting


def bezier_curve(points, num_points=100):
    """Compute the Bezier curve for the given points."""
    n = len(points) - 1
    t = np.linspace(0, 1, num_points)
    curve = np.zeros((num_points, 2))
    for i in range(n + 1):
        binomial_coeff = comb(n, i)
        curve += binomial_coeff * \
            ((1 - t) ** (n - i))[:, np.newaxis] * \
            (t ** i)[:, np.newaxis] * points[i]
    return curve

# Fit Ellipse


def fit_ellipse(points):
    """Fit an ellipse to the given points using skimage."""
    ellipse = EllipseModel()
    if ellipse.estimate(points):
        return ellipse.params  # (xc, yc, a, b, theta)
    else:
        return None

# Validate and standardize color


def validate_color(color):
    """Validate and standardize color values."""
    try:
        if isinstance(color, (int, float)):
            return 'C0'  # Default color
        if isinstance(color, str):
            if color.startswith('#') or color in plt.colors.CSS4_COLORS:
                return color
            # If not valid, return a default color
        return 'C0'
    except (ValueError, AttributeError):
        return 'C0'

# Polynomial Fitting


def fit_polynomial(points, degree=2):
    """Fit a polynomial of a given degree to the points."""
    X = points[:, 0]
    y = points[:, 1]
    poly_coeffs = np.polyfit(X, y, degree)
    poly = np.poly1d(poly_coeffs)
    x_range = np.linspace(np.min(X), np.max(X), 100)
    y_fit = poly(x_range)
    return np.column_stack((x_range, y_fit))

# Plotting


def plot_shapes(data):
    """Plot the shapes from the data."""
    plt.figure(figsize=(8, 8))
    for shape_num in np.unique(data[:, 0]):
        shape_data = data[data[:, 0] == shape_num]
        color = validate_color(shape_data[0, 1])

        # Ensure data is numeric
        try:
            points = shape_data[:, 2:].astype(float)  # Convert to float
        except ValueError as e:
            print(f"Error converting shape data to float: {e}")
            continue  # Skip this shape if conversion fails

        # Noise Removal & Regularization
        points = rdp(points, epsilon=2.0)

        # Check if points form a straight line
        if is_potential_line(points):
            # Convert to perfect straight line
            straight_line = make_straight_line(points)
            plt.plot(straight_line[:, 0], straight_line[:, 1],
                     color=color, label=f'Shape {shape_num}')
        else:
            # Apply Bezier Curve Smoothing
            smoothed_points = bezier_curve(points)
            plt.plot(smoothed_points[:, 0], smoothed_points[:, 1],
                     color=color, label=f'Shape {shape_num}')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.show()

# Main process function


def process_shapes(csv_file):
    """Process and plot shapes from a CSV file."""
    data = load_csv(csv_file)
    plot_shapes(data)


# Example usage
csv_file = 'frag0.csv'
process_shapes(csv_file)
