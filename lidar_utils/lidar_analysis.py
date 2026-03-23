import numpy as np

def simple_corners(points, window=3):
	"""
	Takes the list of boundary points.
	Returns a list of (x, y) coordinates that are local depth maxima over a
	window of neighbours on each side, and whose average spatial distance to
	those neighbours is below the proximity threshold.
 
	window: number of neighbours to consider on each side (default 3).
	"""
	corners = []
	n = len(points)
	for i in range(window, n - window):
		x, y = points[i]
		d = (x**2 + y**2) ** 0.5
 
		# Depth check: must be further than every neighbour in the window
		if not all(
			d > (points[i + j][0]**2 + points[i + j][1]**2) ** 0.5
			for j in range(-window, window + 1) if j != 0
		):
			continue
 
		# Proximity check: average spatial distance to all window neighbours
		avg_dist = sum(
			((x - points[i + j][0])**2 + (y - points[i + j][1])**2) ** 0.5
			for j in range(-window, window + 1) if j != 0
		) / (2 * window)
		if avg_dist >= 0.1:
			continue
 
		corners.append(points[i])
	return corners
