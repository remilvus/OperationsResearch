from time import perf_counter


class Timer:
    def __init__(self):
        self.time = perf_counter()

    def start(self):
        self.time = perf_counter()

    def stop(self, name=''):
        print(name, "time:", perf_counter() - self.time)
        self.start()

    def __call__(self):
        self.start()
