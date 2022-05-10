import numpy as np
import sys
from math import floor

INPUT_PATH = sys.argv[1]
OUTPUT_PATH = sys.argv[2]
# INPUT_PATH = './Assignment2_sampletestcases/TestCases/Q4/input.txt'
# OUTPUT_PATH = './Assignment2_sampletestcases/TestCases/Q4/output_pred.txt'
EPSILON = 1e-12

def read_input(file_path):
    with open(file_path, 'r') as f:
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

def write_output(output_file_path, value, x, solved, status):
    with open(output_file_path, 'w') as f:
        if status == 'U':
            print('Unbounded', file=f)
        elif status == 'I':
            print('Infeasible', file=f)
        else:
            print(np.round(value, 6), file=f)
            final_x = ' '.join(map(str, x))
            print(final_x, file=f)
        print(solved, file=f)

A, b, c = read_input(INPUT_PATH)
SLACK_VAR_COUNT = b.shape[0]
ACTUAL_VAR_COUNT = A.shape[1]
VAR_COUNT = A.shape[1]  
ARTIFICIAL_VARS = []

def get_objective_row(Ab, An, cb, cn):
    Ab_inv = np.linalg.inv(Ab)
    y = Ab_inv @ An
    k = (cb.T @ y) - cn.T
    return k, y
    
def construct_tableau(basis, non_basic, Ab, An, rhs, cb, cn):
    tableau = np.zeros((len(basis) + 1, len(basis) + len(non_basic) + 2))
    k, y = get_objective_row(Ab, An, cb, cn)
    tableau[0, 0], tableau[0, -1] = 1, cb.T @ rhs

    for i in range(len(basis)):
        tableau[i + 1, basis[i] + 1] = 1
    for j in range(1, len(basis) + 1):
        tableau[j, -1] = rhs[j - 1]
    for i in range(len(non_basic)):
        idx = non_basic[i]
        tableau[1:, idx + 1] = y[:, i]
        tableau[0, idx + 1] = k[i] 
    return tableau

def get_phase_1_starter(A):
    global ARTIFICIAL_VARS, SLACK_VAR_COUNT, VAR_COUNT
    basis, done_rows = [0] * A.shape[0], set()
    I = np.identity(A.shape[0])
    
    for i in range(len(ARTIFICIAL_VARS)):
        basis[ARTIFICIAL_VARS[i]] = VAR_COUNT + SLACK_VAR_COUNT + i
        done_rows.add(ARTIFICIAL_VARS[i])
       
    for i in range(A.shape[1]):
        for j in range(A.shape[0]):
            if (A[:, i] == I[:, j]).all() and j not in done_rows:
                basis[j] = i
                done_rows.add(j) 
    return basis
       
def pivot_operations(tableau, r, j):
    m = tableau.shape[0]
    tableau[r, :] /= tableau[r, j]
    for i in range(m):
        if i == r:
            continue
        tableau[i, :] -= (tableau[i, j] * tableau[r, :])
    return tableau

# If there are any negative bs, change it such that it becomes >=, and then add neg slack vars
def add_slack_vars(_A, _b):
    A, b = _A.tolist(), _b.tolist()
    for i in range(len(b)):
        if b[i] < 0:
            A[i] = [-el for el in A[i]]
            A[i] += [-1 if j == i else 0 for j in range(len(b))]
            b[i] *= -1
        else:
            A[i] += [1 if j == i else 0 for j in range(len(b))]
    return np.asarray(A), np.asarray(b)

# insert artificial variables wherever there is neg slack variable, add >= 0 constraints for both slack and artifical vars      
def add_artificial_vars(_A, _b):
    global ARTIFICIAL_VARS
    A, b = _A, _b
    I = np.identity(b.shape[0])
    unique_var_present = set()
    for i in range(A.shape[1]):
        for j in range(b.shape[0]):
            if (A[:, i] == I[:, j]).all():
                unique_var_present.add(j)
    ARTIFICIAL_VARS = [r for r in range(A.shape[0]) if r not in unique_var_present]
    new_cols = np.asarray([[1 if i == j else 0 for i in range(A.shape[0])] for j in ARTIFICIAL_VARS]).T
    if new_cols.shape[0] > 0:
        A = np.hstack([A, new_cols])
    return A

def rectify_basis_cols(tableau, basis):
    for i in range(len(basis)):
        if abs(tableau[0, basis[i] + 1]) < EPSILON:
            continue
        k = tableau[0, basis[i] + 1]
        a = tableau[i + 1, basis[i] + 1]
        tableau[0, :] -= ((k / a) * tableau[i + 1, :])
    for i in range(len(basis)):
        tableau[0, basis[i] + 1] = 0
    return tableau
     
def simplex_loop(_tableau, _basis, _non_basic):
    tableau, basis, non_basic = _tableau, _basis, _non_basic
    while True:
        if not np.any(tableau[0, 1:-1] > EPSILON):
            # Termination
            return tableau, tableau[0, -1], basis, tableau[1:, -1], 'S'
        # Bland's rule for handling degeneracy
        first_row = np.greater(tableau[0, 1:-1],  np.asarray([0] * (tableau.shape[1] - 2)))
        if True not in first_row.tolist():
            # Termination
            return tableau, tableau[0, -1], basis, tableau[1:, -1], 'S'
        # Leaving basis            
        j = np.where(first_row == True)[0][0] + 1
        yj = tableau[1:, j]
        if not np.any(yj > EPSILON):
            # Unbounded
            return tableau, tableau[0, -1], basis, tableau[1:, -1], 'U'
        ratio_pairs = []
        for i in range(1, tableau.shape[0]):
            if yj[i - 1] <= EPSILON:
                continue
            ratio_pairs.append((tableau[i, -1] / yj[i - 1], i))
        # Entering basis
        r = sorted(ratio_pairs)[0][1]
        tableau = pivot_operations(tableau, r, j)
        _idx = non_basic.index(j - 1)
        basis[r - 1], non_basic[_idx] = non_basic[_idx], basis[r - 1]   
               
def simplex(A, b, c, _basis, _rhs):
    tableau = np.asarray([])
    basis, non_basic = _basis, [i for i in range(A.shape[1]) if i not in _basis]
    Ab, An = A[:, basis], A[:, non_basic]
    init_shape = c.shape[0]
    c = np.concatenate([c, np.asarray([0] * (A.shape[1] - init_shape))])
    cb, cn, rhs = c[basis], c[non_basic], _rhs
    tableau = construct_tableau(basis, non_basic, Ab, An, rhs, cb, cn)
    return simplex_loop(tableau, basis, non_basic)

def construct_phase2_tableau(A, c, _tableau, basis):
    global VAR_COUNT, ARTIFICIAL_VARS, SLACK_VAR_COUNT
    # Transforming such that the basis variables have zeroes in objective row
    tableau = rectify_basis_cols(_tableau, basis)
    # Deleting artificial variables rows and columns
    cols_to_delete = [i for i in range(1 + VAR_COUNT + SLACK_VAR_COUNT, tableau.shape[1] - 1)]
    rows_to_delete = [i for i in range(len(basis)) if basis[i] >= (SLACK_VAR_COUNT + VAR_COUNT)]
    new_basis = [basis[i] for i in range(len(basis)) if i not in rows_to_delete]
    new_non_basic = [i for i in range(VAR_COUNT + SLACK_VAR_COUNT) if i not in new_basis]
    tableau = np.delete(tableau, cols_to_delete, axis=1)
    tableau = np.delete(tableau, rows_to_delete, axis=0)
    # Reverting to the previous objective function
    # print(new_basis, new_non_basic)
    # print(new_basis, new_non_basic)
    Ab, An = A[:, new_basis], A[:, new_non_basic]
    cb, cn = c[new_basis], c[new_non_basic]
    tableau[0, -1] = 0
    k, y = get_objective_row(Ab, An, cb, cn)
    for i in range(len(new_non_basic)):
        idx = new_non_basic[i]
        tableau[0, idx + 1] = k[i] 
    return tableau, new_basis        
               
def construct_x(basis, values, with_slack=False):
    global ACTUAL_VAR_COUNT, SLACK_VAR_COUNT
    x = [0] * ACTUAL_VAR_COUNT
    if with_slack:
        x = [0] * (ACTUAL_VAR_COUNT + SLACK_VAR_COUNT)
    for i in range(len(basis)):
        if basis[i] < ACTUAL_VAR_COUNT:
            x[basis[i]] = values[i]
        if with_slack and basis[i] < (VAR_COUNT + SLACK_VAR_COUNT):
            x[basis[i]] = values[i]
    return np.asarray(x)

def is_done(tableau, basis):
    global ACTUAL_VAR_COUNT
    for i in range(len(basis)):
        if basis[i] < ACTUAL_VAR_COUNT:
            if tableau[i + 1, -1] - int(tableau[i + 1, -1]) > EPSILON:
                return False
    return True

def two_phase_simplex(_A, _b, c, add_slack=False):
    global VAR_COUNT, SLACK_VAR_COUNT, ARTIFICIAL_VARS
    if add_slack:
        A, b = add_slack_vars(_A, _b)
    else:
        A, b = _A, _b
    SLACK_VAR_COUNT = A.shape[1] - _A.shape[1]
    VAR_COUNT = _A.shape[1]
    A_art = add_artificial_vars(A, b)
    # print(A_art, b, c)
    basis = get_phase_1_starter(A_art)
    # print(basis)
    # print(A_art)
    # print(SLACK_VAR_COUNT, ARTIFICIAL_VARS, VAR_COUNT)
    if len(ARTIFICIAL_VARS) > 0:
        new_c = [0] * (SLACK_VAR_COUNT + VAR_COUNT)
        new_c += [1 for i in range(len(ARTIFICIAL_VARS))]
        # print(A_art, b, np.asarray(new_c), basis, b)
        tableau, obj, basis, values, status = simplex(A_art, b, np.asarray(new_c), basis, b)
        # print(tableau, obj, basis, values, status)
        if abs(obj) > EPSILON:
            # Infeasible
            return np.asarray([[]]), 0, [], [], 'I'
        c = np.concatenate([c, np.asarray([0] * SLACK_VAR_COUNT)])
        tableau, basis = construct_phase2_tableau(A, c, tableau, basis)
        non_basic = [i for i in range(SLACK_VAR_COUNT + VAR_COUNT) if i not in basis]
        t, obj, basis, values, status = simplex_loop(tableau, basis, non_basic)
        return t, obj, basis, values, status
    else:
        # Directly going to second step as artificial variables are not needed (trivial basis exists)
        c = np.concatenate([c, np.asarray([0] * SLACK_VAR_COUNT)])
        t, obj, basis, values, status = simplex(A, b, c, basis, b)
        return t, obj, basis, values, status

def update_constraints(_A, _b, tableau, idx):
    row, val = tableau[idx + 1, 1:-1], tableau[idx + 1, -1]
    # new_row = -1 * (row - np.floor(row))
    # new_val = -1 * (val - floor(val))
    new_row = np.floor(row)
    new_row = np.append(new_row, 1)
    new_val = floor(val)
    new_col = np.array([0] * tableau[1:, :].shape[0])
    b = np.append(_b, new_val)
    A = np.column_stack((_A, new_col))
    A = np.vstack([A, new_row])
    # new_row = np.append(new_row, new_val)
    # tableau = np.vstack([tableau, new_row])
    # print(tableau)
    return A, b
      
def gomory_cut(_A, _b, _c):
    global SLACK_VAR_COUNT
    A, b, c = _A, _b, -1 * _c
    _vars = 0
    solved = 0
    tableau, basis, non_basic = np.array([]), [], []
    while True:
        tableau, obj, basis, values, status = two_phase_simplex(A, b, c, add_slack = solved == 0)
        # print(tableau)
        if solved == 0:
            c = np.concatenate([c, np.asarray([0] * SLACK_VAR_COUNT)])
            A, b = add_slack_vars(A, b)
        # print(values)
        if len(tableau) == 0:
            return -1.0, [], solved, 'I'            
        _vars = tableau.shape[0] - 2
        # print(values)
        solved += 1
        if is_done(tableau, basis):
            x = construct_x(basis, values)
            return obj, x.astype(int), solved, 'S'            
        if status != 'S':
            return -1.0, [], solved, status
        frac_values = [np.round(values[i] - int(values[i]), 6) for i in range(len(values)) if basis[i] < ACTUAL_VAR_COUNT]
        if len(frac_values) == 0:
            x = construct_x(basis, values)
            return obj, x.astype(int), solved, 'S'           
        i = np.argmax(np.asarray(frac_values))
        if frac_values[i] < EPSILON:
            x = construct_x(basis, values)
            return obj, x.astype(int), solved, 'S'
        A, b = update_constraints(A, b, tableau, i)
        c = np.append(c, 0)
        # basis.append(_vars)
        # _vars += 1
        # non_basic = [i for i in range(_vars) if i not in basis]

obj, x, solved, status = gomory_cut(A, b, c)
write_output(OUTPUT_PATH, round(obj, 6), x, solved, status)
