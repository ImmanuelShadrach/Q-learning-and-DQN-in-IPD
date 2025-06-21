import axelrod as axl
def check(history, targets=((axl.Action.C, axl.Action.D), (axl.Action.D, axl.Action.C))):
    max_len = 0
    max_start = None

    for start in range(len(history) - 1):  # need at least one pair to check alternation
        tolerance = 0
        streak_len = 1
        prev = history[start]

        if prev not in targets:
            continue  # skip if starting point isn't (C, D) or (D, C)

        for i in range(start + 1, len(history)):
            current = history[i]
            expected_next = targets[1] if prev == targets[0] else targets[0]

            if current == expected_next:
                streak_len += 1
                prev = current
            else:
                tolerance += 1
                allowed_tolerance = streak_len // 10
                if tolerance > allowed_tolerance:
                    break
                else:
                    streak_len += 1
                    prev = current  # update even if noisy

        if streak_len > max_len:
            max_len = streak_len
            max_start = start

    return (max_start, max_len) if max_start is not None else (None, 0)


def check(history, target=(axl.Action.D, axl.Action.D)):
    max_len = 0
    max_start = None

    for start in range(len(history)):
        count = 0
        tolerance = 0

        for i in range(start, len(history)):
            total_so_far = i - start + 1
            current = history[i]

            # Adaptive allowed tolerance = 10% of current streak length (rounded down)
            allowed_tolerance = total_so_far // 10

            if current == target:
                count += 1
            else:
                tolerance += 1
                if tolerance > allowed_tolerance:
                    break  # Too noisy, break this streak

            # Update if this streak beats the max
            if total_so_far > max_len:
                max_len = total_so_far
                max_start = start

    return (max_start, max_len) if max_start is not None else (None, 0)



def check(history, target=(axl.Action.C, axl.Action.C)):
    max_len = 0
    max_start = None

    for start in range(len(history)):
        count = 0
        tolerance = 0

        for i in range(start, len(history)):
            total_so_far = i - start + 1
            current = history[i]

            # Adaptive allowed tolerance = 10% of current streak length (rounded down)
            allowed_tolerance = total_so_far // 10

            if current == target:
                count += 1
            else:
                tolerance += 1
                if tolerance > allowed_tolerance:
                    break  # Too noisy, break this streak

            # Update if this streak beats the max
            if total_so_far > max_len:
                max_len = total_so_far
                max_start = start

    return (max_start, max_len) if max_start is not None else (None, 0)