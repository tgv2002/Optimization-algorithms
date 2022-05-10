import numpy as np
from math import ceil, log, sqrt

INPUT_PATH = sys.argv[1]
OUTPUT_PATH = sys.argv[2]

def read_input(input_file_path):
    with open(input_file_path, 'r') as f:
        try:
            lines = [line.strip() for line in f]
            other_lines = ['start A', 'end A', 'start b', 'end b', 'start c', 'end c']
            other_lines_f = [lines[0], lines[-7], lines[-6], lines[-4], lines[-3], lines[-1]]
            c, b = list(map(float, lines[-2].split())), list(map(float, lines[-5].split()))
            A = [list(map(float, line.split())) for line in lines[1:-7]]
            if len(b) != len(A) or len(c) != len(A[0]) or other_lines != other_lines_f:
                print('Invalid input parameters')
                sys.exit()                
        except:
            print('Invalid input parameters')
            sys.exit()
    return np.asarray(A), np.asarray(b), np.asarray(c)

def get_thresholds_and_initialization(A, b, c):
    n, U = len(c), max(np.amax(A), np.amax(b))
    x0 = np.asarray([0.0] * n).reshape(1, -1).T
    v = (n ** (-n)) * ((n * U) ** (-n * n * (n + 1)))
    V = ((2 * n) ** n) * ((n * U) ** (n * n))
    Dt = (n * ((n * U) ** (2 * n))) * np.identity(n)
    return v, V, x0, Dt

def ellipsoid(A, b, c, obj_A, obj_b):
    n = len(c)
    print(obj_A, obj_b)
    if obj_A.shape[0] == 0:
        v, V, x0, Dt = get_thresholds_and_initialization(A, b, c)
    else:
        v, V, x0, Dt = get_thresholds_and_initialization(np.concatenate((A, obj_A), axis=0), np.concatenate((b, obj_b), axis=0), c)
    t, time_steps, xt = 0, int(ceil(2 * (n + 1) * log(V / v))), x0
    print(time_steps)
    while t <= time_steps:
        if t == time_steps:
            return 0, np.asarray([-1] * n), "Infeasible"
        constraint_res = np.less_equal(A @ xt,  b)
        if obj_A.shape[0] > 0:
            constraint_res2 = np.less(obj_A @ xt, obj_b)
        else:
            constraint_res2 = []
        if False not in constraint_res and False not in constraint_res2:
            return np.dot(c, xt), xt, "solved"
        if False in constraint_res:
            i = -1
            for _k in range(constraint_res.shape[0]):
                if False in constraint_res[_k]:
                    i = _k
                    break
            ai = A[i, :].reshape(1, -1).T
        else:
            i = -1
            for _k in range(constraint_res.shape[0]):
                if False in constraint_res[_k]:
                    i = _k
                    break
            ai = obj_A[i, :].reshape(1, -1).T
        deno = ai.T @ Dt @ ai
        # if deno < 0:
        #     return 0, np.asarray([-1] * n), "Infeasible"   
        term_x = (Dt @ ai) / sqrt(abs(deno))
        term_D = (Dt @ ai @ ai.T @ Dt) / deno
        xt = xt + ((1 / (n + 1)) * term_x)
        Dt = ((n**2) / (n**2 - 1)) * (Dt - ((2 / (n + 1)) * term_D))
        print('ai=', i, 'xt=', xt)
        t += 1

def write_output(output_file_path, value, x):
    with open(output_file_path, 'w') as f:
        if x[0] == -1:
            print('Infeasible', file=f)
        else:
            print(value.tolist()[0], file=f)
            final_x = ' '.join(map(str, [el[0] for el in x.tolist()]))
            print(final_x, file=f)
            
def sliding_objective(A, b, c, output_file_path):
    obj_A, obj_b = np.asarray([]), np.asarray([])
    results, prev_results = [], []
    while True:
        results = ellipsoid(A, b, c, obj_A, obj_b)
        print(results)
        if results[-1] == "Infeasible":
            write_output(output_file_path, prev_results[0], prev_results[1])
            return
        if obj_A.shape[0] == 0:
            obj_A = np.asarray([c])
        else:
            obj_A = np.vstack([obj_A, c])
        obj_b = np.append(obj_b, results[0])
        prev_results = results
    
A, b, c = read_input(INPUT_PATH)
for j in range(len(A[0])):
    A = np.vstack([A, np.asarray([-1.0 if k == j else 0.0 for k in range(len(A[0]))])])
    b = np.append(b, 0.0)
print(A)
print(b)
sliding_objective(A, b, c, OUTPUT_PATH)