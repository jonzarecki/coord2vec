from datetime import timedelta, datetime

from typing import List, Tuple


def get_batch_time_intervals(start_time: datetime, end_time: datetime, batch_size_mins: int) -> List[Tuple[datetime, datetime]]:
    """
    Returns a list of datetime tuples, each representing a batch for the specified range, with $step_size_mins steps in minutes
    Args:
        start_time: datetime with the starting time for the range
        end_time: datetime with the ending time for the range
        batch_size_mins: the step size for the range in minutes

    Returns:
        The range list of datetime objects (tuples of (start_time, end_time)
    """
    time_intervals = []
    batch_st_time = start_time
    while batch_st_time < end_time:

        batch_end_time = batch_st_time + timedelta(minutes=batch_size_mins)
        if batch_end_time > end_time:
            batch_end_time = end_time

        time_intervals.append((batch_st_time, batch_end_time))

        batch_st_time = batch_end_time

    return time_intervals
