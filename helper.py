def array_to_count(array):
    element_counts = {item: array.count(item) for item in set(array)}
    dict_as_custom_string = ', '.join([f'{k}: {v}' for k, v in element_counts.items()])
    return dict_as_custom_string


