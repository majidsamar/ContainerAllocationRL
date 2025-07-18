from docplex.mp.model import Model
import numpy as np
from datetime import datetime

# m = Model(name='test')
# x = m.binary_var(name='x')
# y = m.binary_var(name='y')
# m.add_constraint(x + y <= 1)
# m.maximize(x + 2*y)
# solution = m.solve()
# print(solution)


# model = Model(name="simple_LP")
# x = model.continuous_var(name="x")
# y = model.continuous_var(name="y")
#
# model.add_constraint(2*x + 3*y <= 12, "c1")
# model.add_constraint(x + y <= 5, "c2")
#
# model.maximize(3*x + 5*y)
# solution = model.solve()
#
# if solution:
#     print("Objective value:", solution.objective_value)
#     print("x =", x.solution_value)
#     print("y =", y.solution_value)
# else:
#     print("No solution found")





def run_solver(num_containers,num_stacks,stack_height,dwell_times,time_limit_sec):
    C = range(num_containers)
    S = range(num_stacks)
    T = range(stack_height)

    # Model
    mdl = Model("container_stacking_fifo")
    mdl.parameters.timelimit = time_limit_sec
    # Binary decision variables:
    # x[c,s,t] = 1 if container c is placed at stack s, tier t
    x = {(c, s, t): mdl.binary_var(name=f"x_{c}_{s}_{t}") for c in C for s in S for t in T} # 30 * 6 * 5 = 900 only for this

    # Constraint 1: each container must be assigned to one and only one position
    for c in C:
        mdl.add_constraint(mdl.sum(x[c, s, t] for s in S for t in T) == 1)
        # container c can not be located in two places

    # constraint 2: maximum one container per position
    for s in S:
        for t in T:
            mdl.add_constraint(mdl.sum(x[c, s, t] for c in C) <= 1)

    #constraint 3: keeping stacking order (no floating on the air container)
    for s in S:
        for t in range(1, stack_height):
            # here sum applies on many zero and only one 1 : [0,0,0,1,0,0]
            mdl.add_constraint(
                mdl.sum(x[c, s, t] for c in C) <= mdl.sum(x[c, s, t-1] for c in C)
            )

    # constraint 4:  FIFO ordering ,
    # using the fact that:
    # if j>i which means container j is after container i, and if t2 > t1 which means
    # container i is higher than container j,
    # then both can not ba in the same stack (if they are then x[j, s, t1] + x[i, s, t2] == 2 and FIFO is violated)
    # this ensures that their stack is different

    for s in S:
        for t1 in range(stack_height):
            for t2 in range(t1 + 1, stack_height):
                for i in C:
                    for j in C:
                        if j > i:
                            mdl.add_constraint(x[j, s, t1] + x[i, s, t2] <= 1)

    # defining violation variables: v[c1, c2] = 1 if c1 is above c2 and has higher dwell
    violations = []
    for s in S:
        for t1 in range(1, stack_height):
            for t2 in range(t1):
            # for t2 in range(t1-1,t1):  # only consider one tier before c1
                for c1 in C:
                    for c2 in C:
                        if dwell_times[c1] >= dwell_times[c2]:
                            v = mdl.binary_var(name=f"v_{c1}_{c2}_{s}_{t1}_{t2}")
                            mdl.add_constraint(
                                v >= x[c1, s, t1] + x[c2, s, t2] - 1 # -1 to model logical and
                                # c1 has higher dwell time that c2 but sits on t1 which is higher than t2
                                # if both x[c1,s,t1] == 1 and x[c2,s,t2] == 1 --> then v must be 1
                                # otherwise, v can be 0
                            )
                            violations.append(v)


    mdl.minimize(mdl.sum(violations))
    solution = mdl.solve(log_output=True)


    allocation = {}
    if solution:
        print("optimal allocation with minimum violations:")
        stacks = {s: [None] * stack_height for s in S}
        for c in C:
            for s in S:
                for t in T:
                    if x[c, s, t].solution_value > 0.5:
                        stacks[s][t] = c

        for s in S:
            print(f"\nStack {s}:")
            for t in reversed(T):
                c = stacks[s][t]
                if c is not None:
                    print(f"  Tier {t}: Container {c} (Dwell {dwell_times[c]})")
                else:
                    print(f"  Tier {t}: Empty")

        print("\nTotal dwell time violations:", int(mdl.objective_value))
    else:
        print("No solution found.")
    print("Dwell times:", dwell_times)


num_containers = 135
num_stacks = 27
stack_height = 5
time_limit_sec = 300 # 5 min


# np.random.seed(42)


dwell_times = np.random.randint(1, 21, size=num_containers)
# dwell_times = [2,1,3,8,5,4,3,7,6]
print("Dwell times:", dwell_times)

start_time = datetime.now()
run_solver(num_containers=num_containers,
           num_stacks=num_stacks,
           stack_height=stack_height,
           dwell_times=dwell_times,
           time_limit_sec= time_limit_sec)
print("\n\nDuration:", datetime.now() - start_time)