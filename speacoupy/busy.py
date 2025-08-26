from __future__ import annotations
import sys, threading, time

class _SpinnerThread(threading.Thread):
	def __init__(self, message: str):
		super().__init__(daemon=True)
		self.message = message.rstrip()
		self._running = threading.Event()
		self._running.set()
		self._frames = ['⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏']
		self._i = 0

	def stop(self):
		self._running.clear()

	def run(self):
		# print first line
		stderr = sys.stderr
		while self._running.is_set():
			frame = self._frames[self._i % len(self._frames)]
			self._i += 1
			stderr.write(f"\r{self.message} {frame}")
			stderr.flush()
			time.sleep(0.15)
		# clear spinner and finalize line
		stderr.write(f"\r{self.message} done.\n")
		stderr.flush()

class busy:
	"""Context manager that shows a simple spinner w/o numbers.
	Always verbose; writes to stderr; no TTY checks.
	Use:
		with busy("Solving…"):
			foo()
	"""
	def __init__(self, message: str):
		self._msg = message
		self._thr = None
	def __enter__(self):
		self._thr = _SpinnerThread(self._msg)
		self._thr.start()
		return self
	def __exit__(self, exc_type, exc, tb):
		if self._thr is not None:
			self._thr.stop()
			self._thr.join(timeout=1.0)
		# do not suppress exceptions
		return False
