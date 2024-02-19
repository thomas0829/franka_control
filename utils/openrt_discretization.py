def inverse_discretize_bins(binned_data, min_value, max_value, num_bins=256):
    bin_size = (max_value - min_value) / num_bins
    # Map each bin index to the lower bound of the corresponding value range
    original_data = (int(binned_data) * bin_size) + min_value
    return original_data


def inverse_discretize(action_bins, min_max_lst):
    action_values = action_bins.split(" ")[1:7]
    new_action_values = []
    for i in range(len(action_values)):
        new_action_values.append(
            inverse_discretize_bins(
                action_values[i],
                min_value=min_max_lst[i][0],
                max_value=min_max_lst[i][1],
            )
        )
    return new_action_values
