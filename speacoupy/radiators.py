from collections import deque

def _is_terminal_radiator(el) -> bool:
	# Terminal radiator: object that radiates by itself
	# Heuristic: has Sd+impedance (e.g. RadiationPiston) or exposes Z_rad
	return (hasattr(el, "Sd") and hasattr(el, "impedance")) or hasattr(el, "Z_rad")

def collect_radiators(start_element):
	"""Traverse from start_element and return list of (label, element) for terminal radiators."""
	visited = set()
	found = []
	queue = deque([start_element])

	while queue:
		el = queue.popleft()
		if id(el) in visited:
			continue
		visited.add(id(el))

		if _is_terminal_radiator(el):
			lbl = getattr(el, "label", str(id(el)))
			found.append((lbl, el))
			# Do not traverse past a terminal radiator
			continue

		# Traverse common containers
		if hasattr(el, "elements"):
			for sub in getattr(el, "elements", []):
				queue.append(sub)

		for attr in ("front_load", "back_load", "port", "mouth_radiator"):
			if hasattr(el, attr):
				sub = getattr(el, attr)
				if sub is not None:
					queue.append(sub)

	return found

