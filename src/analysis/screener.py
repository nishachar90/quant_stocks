def filter_stocks(
    data,
    column,
    minimum=None,
    maximum=None,
):
    filtered = data.copy()

    if minimum is not None:
        filtered = filtered[filtered[column] >= minimum]

    if maximum is not None:
        filtered = filtered[filtered[column] <= maximum]

    return filtered
