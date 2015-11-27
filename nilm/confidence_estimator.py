import itertools

def confidence(data):
    # If we observe nothing about the data,
    # then we have no confidence in our estimate.
    if (len(data) == 0): return float("infinity")

    mean = sum(data)/len(data)
    variance = sum((x - mean)**2 for x in data)/n

    return variance/len(data)





def only_device(device, on_off, time, devices):
    devices_on = {x for x in devices if on_off[(time, x)]}

    return (device in devices_on) and (len(devices_on) == 1)

def sort_data(devices, time_series, on_off):
    data = {x:[] for x in devices}

    for (device, time) in itertools.product(devices, time_series.keys()):
        if only_device(device, on_off, time, devices):
            data[device].append(time_series[time])

    return data


def changed_devices(devices, on_off, i):
    return [x for x in devices if on_off[i, x] != on_off[i-1, x]]

def get_changed_data(time_series, on_off, devices):
    data = {x:[] for x in devices}

    for i in range(2,len(time_series)):
        changed = changed_devices(devices, on_off, i)

        if (len(changed) == 1):
            data[changed[0]].append(time_series[i] - time_series[i-1])





# Given a time series of data (a dictionary mapping time to power),
# with on and off indicators (dictionary of (time, device #) pairs,
# computes the best time estimators by the confidence interval method.
# Assumes that the time data between the series and indicators is the
# same, but does not assume how it is distributed. This programs
# assumes that every device will be able to be calculated at some
# point. If not, this function not be able to estimate the programs accurately
def confidence_estimator(time_series, on_off, devices, data_sorter):
    if (len(devices) == 0): return {}

    data = data_sorter(devices, time_series, on_off)

    # Pick data to remove according to some heuristic
    heuristic = lambda x: confidence(data[x])
    choice = min(devices, key = heuristic)

    if (heuristic(remove_choice) == float('infinity')):
        # Need to pick a better approach, try
        # generating more data using level technique.
        print("Not enough data")
        mean_choice = 0

    else: mean_choice = sum(data)/len(data)

    new_devices = [x for x in devices if x is not choice]
    new_series = {time: time_series[time] - is_on[time, choice]*mean_choice
        for time in time_series.keys()}

    calculated_means = signal_confidence_estimator(new_series, on_off, new_devices, data_sorter)
    calculated_means[choice] = mean_choice

    return calculated_means





# Obtains estimates of time series based on immediate
# changes in energy levels corresponding to a device
# being switched on while other devices do not change.
# In this code, the time in the time_series and on_off
# indicators is assumed to be an integer list.
def level_estimate(time_series, on_off, devices):
    estimate = {}
    while (len(estimate.keys()) < len(devices)):
        data = get_changed_data(time_series, on_off, devices)

        for device in devices:
            if len(data[device]) != 0:
                if (device not in estimate.keys()):
                    estimate[device] += data[device]

                # Turn off device after we have learnt it
                for i in range(1, time_series.keys()):
                    time_series[i] = time_series[i] - on_off[i, device]*estimate[device]
                    on_off[i, device] = False

    return estimate

time_series = {1:5, 2:2, 3:4}
on_off = {(1,'a'): True, (2,'a'): True, (3,'a'): True, (1, 'b'): False, (2,'b'): True, (3,'b'): False}
devices = ['a','b']

print(level_estimate(time_series, on_off, devices))