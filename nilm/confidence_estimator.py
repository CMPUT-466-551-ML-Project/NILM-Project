import itertools


def confidence(data):
    """ A Heuristic for how usable our current estimate of data is """

    if len(data) == 0:
        return float("infinity")

    mean = sum(data)/len(data)
    variance = sum((x - mean)**2 for x in data)/len(data)

    return variance/len(data)


def only_device(device, on_off, time, devices):
    """
    Returns True if the device is the only device active at a certain time.
    """

    devices_on = {x for x in devices if on_off[(time, x)]}

    return (device in devices_on) and (len(devices_on) == 1)


def sort_data(devices, time_series, on_off):
    """
    Generates usable samples for each device, where that
    device was the only device active at a single time period.
    """

    data = {x:[] for x in devices}

    for (device, time) in itertools.product(devices, time_series.keys()):
        if only_device(device, on_off, time, devices):
            data[device].append(time_series[time])

    return data


def changed_devices(devices, on_off, i):
    """
    Returns all devices whose I/O state changed between at a single time period
    i.
    """
    return [x for x in devices if on_off[i, x] != on_off[i-1, x]]


def get_changed_data(devices, time_series, on_off):
    """
    Generates data for each device by the step inference method, calculating
    the change in energy usage as a single device changes.
    """

    data = {x:[] for x in devices}

    for i in range(2, len(time_series)):
        changed = changed_devices(devices, on_off, i)

        if len(changed) == 1:
            data[changed[0]].append(time_series[i] - time_series[i-1])

    return data


def confidence_estimator(time_series, on_off, devices, data_sorter):
    """
    Given a time series of data (a dictionary mapping time to power), with
    on/off indicators (dictionary of (time, device) pairs mapping to power,
    computes the best time estimators by the confidence interval subtraction
    method.  Data obtained for the confidence interval measure is obtained via
    the data_sorter function, which is currently implemented to either take
    samples from immediate changes in power from a single device switching on
    (get_changed_data) or taking samples from when a single device is on
    (sort_data).  The program assumes that the time data between the series and
    indicators is the same, but does not assume how it is distributed. This
    programs assumes that every device will be able to be calculated at some
    point. If not, this function is not be able to estimate the programs
    accurately.  get_changed_data assumes that the data is linearly ordered
    (time is integer valued), whereas sort_data can have any time
    representation.
    """

    if len(devices) == 0:
        return {}

    data = data_sorter(devices, time_series, on_off)

    # Pick data to remove according to some heuristic
    heuristic = lambda x: confidence(data[x])
    choice = min(devices, key=heuristic)

    if heuristic(choice) == float('infinity'):
        # Need to pick a better approach, try
        # generating more data using level technique.
        print "Not enough data"
        mean_choice = 0
    else:
        mean_choice = sum(data[choice])/len(data)

    new_devices = [x for x in devices if x is not choice]
    new_series = {time: time_series[time] - on_off[time, choice]*mean_choice
                  for time in time_series.keys()}

    calculated_means = confidence_estimator(new_series, on_off, new_devices,
                                            data_sorter)
    calculated_means[choice] = mean_choice

    return calculated_means
