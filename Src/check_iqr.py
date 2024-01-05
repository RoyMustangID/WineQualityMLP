def check_iqr(data, variable, distance=1.5):
    iqr = data[variable].quantile(0.75) - data[variable].quantile(0.25)

    lower_limit = data[variable].quantile(0.25) - (iqr * distance)
    if lower_limit < 0:
        lower_limit = 0
    upper_limit = data[variable].quantile(0.75) + (iqr * distance)
    
    outlier_count = (
        data[
            (data[variable]
            > upper_limit)
            | (data[variable] < lower_limit)
        ]
        .count()
        .values[0]
    )
    
    return upper_limit, lower_limit