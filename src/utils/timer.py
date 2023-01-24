from datetime import datetime


def timeit(func):
    def timefunc(*args, **kwargs):
        start = datetime.now()
        result = func(*args, **kwargs)
        end = datetime.now()
        print("{} time elapsed (hh:mm:ss.ms) {}".format(func.__name__, end - start))
        return result
    return timefunc

@timeit
def somefunc():
    return 1 + sum(range(1, 100000))



class TimeChecker:
    def __init__(self, max_hours=0, max_minutes=0, max_seconds=0):
        """
            save maximum time duration in seconds
            check whether the program exceed maximum time duration
        """
        self._max_time_duration = 3600 * max_hours + 60 * max_minutes + max_seconds
        assert self._max_time_duration > 0, 'max time duration should be greater than 0'
        print(f'max time duration: {self._max_time_duration}')
        self._time_per_iter = 0
        self._check = None

    def start(self):
        # run start() when you want to start timing.
        self._start_time = datetime.now()

    def check(self, toprint=False):
        """
        should be called each epoch to check elapsed time duration.
        :param toprint:
        :return: whether should stop training
        """
        if self._check is None:
            self._check = datetime.now()
            return False
        else:
            now = datetime.now()
            self._time_per_iter = max((now - self._check).seconds, self._time_per_iter)
            tobreak = (((now - self._start_time).seconds + self._time_per_iter) > self._max_time_duration)
            self._check = now
            if toprint or tobreak:
                print(f'time elapsed from start: {now - self._start_time}')
            return tobreak


if __name__ == '__main__':
    import time
    start = datetime.now()
    print(f"start: {start}")
    # print(somefunc())
    time.sleep(3)
    end = datetime.now()
    print(f"end: {end}")
    duration = end - start
    print(type(duration))
    print(f"duration: {duration}")
    print(f"microseconds: {duration.microseconds}")
    print(f"days: {duration.days}")
    print(duration.resolution)
    print(duration.seconds)
    print(type(duration.seconds))
