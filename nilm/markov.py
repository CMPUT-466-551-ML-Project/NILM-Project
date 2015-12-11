def find_means(l, Y, k):
    # memo[m,k] is the minimum value of the optimization we can do over
    # {Y[0],...,Y[m-1]} with k means to use.
    memo = {}

    # means[m,k] is the value of the k means used to obtain the minimum
    # value over the {Y[0],...,Y[m-1]}
    means = {}

    for m in range(1,len(Y)+1):
        means[m,1] = [float(sum(l[j]*Y[j] for j in range(m)))/float(sum(l[j] for j in range(m)))]
        memo[m,1] = sum(l[j]*(Y[j] - means[m,1][0])**2 for j in range(m))

    for k2 in range(2,k+1):
        for m2 in range(1,len(Y)+1):
            if (m2 <= k2):
                # The best option is just to cover all points. If we ever use these values
                # we are probably overfitting our data.
                memo[m2,k2] = 0
                means[m2,k2] = list(set(Y[x] for x in range(m2)))
                continue

            min_value = float('inf')
            best_means = []

            for p in range(1,m2):
                mean_location = float(sum(l[j]*Y[j] for j in range(p,m2)))/float(sum(l[j] for j in range(p,m2)))
                value = memo[p,k2-1] + sum(l[j]*(Y[j] - mean_location)**2 for j in range(p,m2))

                if value < min_value:
                    best_means = means[p,k2-1] + [mean_location]
                    min_value = value

            memo[m2,k2] = min_value
            means[m2,k2] = best_means

    return (memo[len(Y), k], means[len(Y), k])




def only_switch(indicator, device, devices, t):
    switchers = [d for d in devices if indicator[d,t-1] != indicator[d,t]]
    return device in switchers and len(switchers) == 1

def fit_data(power, device, indicator, k, devices):
    # power[time] is power
    # indicator[device, time] is boolean

    current_length = 0
    change = 0
    power_length_pairs = []
    lone_switcher = False

    for t in sorted(power.keys()):
        if (indicator[device, t] == 0 and current_length != 0):
            if (only_switch(indicator, device, devices, t)):
                lone_switcher = True
                change = abs(power[t] - power[t-1])

            if (lone_switcher):
                power_length_pairs.append((change, current_length))
            current_length = 0

        if (indicator[device, t] != 0 and current_length == 0):
            change = abs(power[t] - power[t-1])
            lone_switcher = only_switch(indicator, device, devices, t)
            current_length = 1

        if (indicator[device, t] != 0 and current_length != 0):
            current_length += 1

    if (current_length != 0):
        power_length_pairs.append((change, current_length))
        current_length = 0

    power_length_pairs.sort(key = lambda x: x[0])

    powers = [power_length_pairs[i][0] for i in range(len(power_length_pairs))]
    lengths = [power_length_pairs[i][1] for i in range(len(power_length_pairs))]

    print("POWERS", powers, lengths)

    estimates = find_means(lengths, powers, k)[1]
    print("ESTIMATES", find_means(lengths, powers, k))

    disaggregated = {}
    best_fit = lambda x: min(estimates, key = lambda y: (y - x)**2)
    change = power[0]
    current_length = 1

    first = True
    for t in sorted(power.keys()):
        if first:
            first = False
            print("BEST", best_fit(change))
            disaggregated[t] = min([0, best_fit(change)], key = lambda x: (x - change)**2)
            continue

        elif (indicator[device, t] == 0):
            current_length = 0
            disaggregated[t] = 0

        elif (current_length == 0):
            change = abs(power[t] - power[t-1])
            disaggregated[t] = best_fit(change)
            current_length = 1

        else: disaggregated[t] = best_fit(change)

    return disaggregated

# NOTE: PAD DATA WITH ONE TIME SAMPLE WHERE EVERYTHING IS OFF

#print(find_means([1,2,3,4,2], [0,1,8,8,9], 3))

#power = {0:0, 1:5, 2:3, 3:2, 4:0, 5:2, 6:3}
#indicator = {(1,0): 0, (1,1):1, (1,2):1, (1,3):0, (1,4):0, (1,5):1, (1,6):1,
#             (2,0): 0, (2,1):1, (2,2):0, (2,3):1, (2,4):0, (2,5):0, (2,6):0}
#k = 2

#print(fit_data(power, 2, indicator, k, [1,2]))
