from datetime import datetime


def timeit(func):
    def timefunc(*args, **kwargs):
        start = datetime.now()
        result = func(*args, **kwargs)
        end = datetime.now()
        print("{} time elapsed (hh:mm:ss.ms) {}".format(func.__name__, end - start))
        return result
    return timefunc

# @timeit
def somefunc():
    result = 1
    for i in range(1, 100000):
        result += i
    return result

if __name__ == '__main__':
    import time
    start = datetime.now()
    print("start: {}".format(start))
    # print(somefunc())
    time.sleep(3)
    end = datetime.now()
    print("end: {}".format(end))
    duration = end - start
    print(type(duration))
    print("duration: {}".format(duration))
    print("microseconds: {}".format(duration.microseconds))
    print("days: {}".format(duration.days))
    print(duration.resolution)
    print(duration.seconds)
    print(type(duration.seconds))
