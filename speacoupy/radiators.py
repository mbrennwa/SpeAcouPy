from collections import deque

def _is_terminal_radiator(el) -> bool:
	# Terminal radiator = object that can radiate by itself
	# Heuristic: RadiationPiston (has .impedance and .Sd), or anything with a callable Z_rad
	return (hasattr(el, "Sd") and hasattr(el, "impedance")) or hasattr(el, "Z_rad")

def collect_radiators(start_element, polarity=1):
	"""Traverse from start_element and return list of (label, element, polarity).
	Polarity is +1 for front, -1 for back (propagated along the path).
	Only terminal radiators are returned (e.g., RadiationPiston).
	"""
	visited = set()
	found = []
	queue = deque([(start_element, polarity)])

	while queue:
		el, pol = queue.popleft()
		if el is None or id(el) in visited:
			continue
		visited.add(id(el))

		# Terminal radiator?
		if _is_terminal_radiator(el):
			lbl = getattr(el, "label", None) or getattr(el, "name", None) or el.__class__.__name__
			found.append((lbl, el, pol))
			continue  # Do not traverse past a terminal radiator

		# Series / Parallel networks (if present)
		if hasattr(el, "elements"):
			for sub in getattr(el, "elements", []):
				queue.append((sub, pol))

		# Composite sub-elements commonly used
		for attr in ("front_load", "back_load", "port", "mouth_radiator"):
			sub = getattr(el, attr, None) if hasattr(el, attr) else None
			if sub is None:
				continue
			if attr == "front_load":
				queue.append((sub, +1))
			elif attr == "back_load":
				queue.append((sub, -1))
			else:
				queue.append((sub, pol))

	return found