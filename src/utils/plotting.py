from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer


def plot_elbow_curve(X, model, k):
    visualizer = KElbowVisualizer(model, k=k)
    visualizer.fit(X)
    visualizer.show()
    return visualizer


def plot_silhouette(X, model):
    visualizer = SilhouetteVisualizer(model, colors="yellowbrick")
    visualizer.fit(X)
    visualizer.show()
    return visualizer
