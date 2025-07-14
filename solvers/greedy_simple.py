import numpy as np


# np.random.seed(42)



def calculate_violations(stack, new_dwell):
    for below_dwell in stack:
        if below_dwell <= new_dwell:
            return 1  # violation
    return 0


def one_round(num_containers,bays,rows,tiers):

    dwell_times = np.random.randint(1, 21, size=num_containers)
    # dwell_times =[2, 1, 3, 8, 5, 4, 3, 7, 6]


    # print("Dwell Times (queue order):", dwell_times)

    stacks = [[] for _ in range(bays* rows)]


    total_violations = 0
    for i, dwell in enumerate(dwell_times):
        best_stack = None
        min_violation = float('inf')

        for s_idx, stack in enumerate(stacks):
            if len(stack) < tiers:
                vio = calculate_violations(stack, dwell)
                if vio < min_violation:
                    min_violation = vio
                    best_stack = s_idx

        if best_stack is not None:
            total_violations += calculate_violations(stacks[best_stack], dwell)
            stacks[best_stack].append(dwell)
        else:
            print(f"Container {i} with dwell {dwell} could not be placed (no space).")

    #print("\nFinal Stacks:")
    # for idx, s in enumerate(stacks):
    #     print(f"Stack {idx}: {s}")

    # print("\nTotal Dwell Time Violations:", total_violations)
    return total_violations


def multiple_rounds(bays,rows,tiers,fullness_ratio):
    num_rounds = 1000
    num_containers = round(rows * bays * tiers * fullness_ratio)
    violation_list = []
    for i in range(num_rounds):
        violation_list.append(one_round(num_containers=num_containers, bays=bays,rows=rows,tiers=tiers))
    #print(violation_list)
    print(f"bay : {bays}, tier:{tiers} kpi: {(sum(violation_list) / len(violation_list))/num_containers :.3f} , "
          f"avg of violation:{sum(violation_list) / len(violation_list):.3f} ")



for t in range(6,9):
    for b in range(4,20):
        multiple_rounds(bays=b,rows=1,tiers=t,fullness_ratio= 1)
    print("\n")