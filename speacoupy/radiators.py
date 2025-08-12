
from collections import deque

def collect_radiators(start_element, polarity=1):
	"""Traverse from start_element to find all radiators, tracking polarity.
	Returns list of (label, element, polarity).
	"""
	visited = set()
	found = []

	queue = deque([(start_element, polarity)])
	while queue:
	    element, pol = queue.popleft()
	    if id(element) in visited:
	        continue
	    visited.add(id(element))

	    # Radiator detection: must have Z_rad method
	    if hasattr(element, "Z_rad"):
	        found.append((getattr(element, 'label', str(element)), element, pol))
	        continue

	    # Series or Parallel
	    if hasattr(element, "elements"):
	        for sub in getattr(element, "elements", []):
	            queue.append((sub, pol))

	    # For driver-like with front/back
	    if hasattr(element, "front_load"):
	        queue.append((element.front_load, +1))
	    if hasattr(element, "back_load"):
	        queue.append((element.back_load, -1))

	return found
