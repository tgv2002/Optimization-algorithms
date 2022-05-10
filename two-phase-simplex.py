import numpy as np
import sys

INPUT_PATH = sys.argv[1]
OUTPUT_PATH = sys.argv[2]
# INPUT_PATH = './Assignment2_sampletestcases/TestCases/Q1/input3.txt'
# OUTPUT_PATH = './Assignment2_sampletestcases/TestCases/Q1/output_pred.txt'
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

def write_output(output_file_path, value, x, status):
    with open(output_file_path, 'w') as f:
        if status == 'U':
            print('Unbounded', file=f)
        elif status == 'I':
            print('Infeasible', file=f)
        else:
            print(np.round(value, 6), file=f)
            final_x = ' '.join(map(str, np.round(x, 6)))
            print(final_x, file=f)

A, b, c = read_input(INPUT_PATH)
SLACK_VAR_COUNT = b.shape[0]
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
def add_artificial_vars(A, b):
    global ARTIFICIAL_VARS
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
        # Entering basis            
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
        # Leaving basis
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

def remove_artificial_from_basis(_tableau, _basis):
    global VAR_COUNT, ARTIFICIAL_VARS, SLACK_VAR_COUNT
    tableau, basis = _tableau, _basis
    non_basic = [i for i in range(VAR_COUNT + SLACK_VAR_COUNT + len(ARTIFICIAL_VARS)) if i not in basis]
    for i in range(len(_basis)):
        if _basis[i] < (VAR_COUNT + SLACK_VAR_COUNT):
            continue
        r = i + 1
        # Finding a col with non-zero co-eff for x
        j = [idx for idx in range(1, VAR_COUNT + SLACK_VAR_COUNT + 1) if abs(tableau[r, idx]) > EPSILON][0]
        tableau = pivot_operations(tableau, r, j)
        _idx = non_basic.index(j - 1)
        basis[r - 1], non_basic[_idx] = non_basic[_idx], basis[r - 1]
    return tableau, basis
    
def construct_phase2_tableau(A, c, _tableau, basis, b):
    global VAR_COUNT, ARTIFICIAL_VARS, SLACK_VAR_COUNT
    # Transforming such that the basis variables have zeroes in objective row
    tableau = rectify_basis_cols(_tableau, basis)
    # Removing artificial variable rows from basis
    tableau, new_basis = remove_artificial_from_basis(tableau, basis)
    # Deleting artificial variable columns
    cols_to_delete = [i for i in range(1 + VAR_COUNT + SLACK_VAR_COUNT, tableau.shape[1] - 1)]
    # rows_to_delete = [i for i in range(len(basis)) if basis[i] >= (SLACK_VAR_COUNT + VAR_COUNT)]
    # new_basis = [basis[i] for i in range(len(basis)) if i not in rows_to_delete]
    new_non_basic = [i for i in range(VAR_COUNT + SLACK_VAR_COUNT) if i not in new_basis]
    tableau = np.delete(tableau, cols_to_delete, axis=1)
    # tableau = np.delete(tableau, rows_to_delete, axis=0)
    # print(len(basis))
    # print(len(new_basis), len(new_non_basic))
    # Reverting to the previous objective function
    Ab, An = A[:, new_basis], A[:, new_non_basic]
    cb, cn = c[new_basis], c[new_non_basic]
    tableau[0, -1] = cb.T @ np.linalg.inv(Ab) @ b
    # print(tableau.shape)
    # print(Ab.shape, An.shape, cb.shape, cn.shape)
    k, y = get_objective_row(Ab, An, cb, cn)
    for i in range(len(new_non_basic)):
        idx = new_non_basic[i]
        tableau[0, idx + 1] = k[i] 
    return tableau, new_basis          
               
def construct_x(basis, values, with_slack=False):
    global VAR_COUNT, SLACK_VAR_COUNT
    x = [0] * VAR_COUNT
    if with_slack:
        x = [0] * (VAR_COUNT + SLACK_VAR_COUNT)
    for i in range(len(basis)):
        if basis[i] < VAR_COUNT:
            x[basis[i]] = values[i]
        if with_slack and basis[i] < (VAR_COUNT + SLACK_VAR_COUNT):
            x[basis[i]] = values[i]
    return x
    
def two_phase_simplex(_A, _b, c):
    global VAR_COUNT, SLACK_VAR_COUNT, ARTIFICIAL_VARS
    A, b = add_slack_vars(_A, _b)
    A_art = add_artificial_vars(A, b)
    basis = get_phase_1_starter(A_art)
    if len(ARTIFICIAL_VARS) > 0:
        new_c = [0] * (SLACK_VAR_COUNT + VAR_COUNT)
        new_c += [1 for i in range(len(ARTIFICIAL_VARS))]
        tableau, obj, basis, values, status = simplex(A_art, b, np.asarray(new_c), basis, b)
        if abs(obj) > EPSILON:
            # Infeasible
            write_output(OUTPUT_PATH, obj, [0], 'I')
            return
        tableau, basis = construct_phase2_tableau(A, np.concatenate([c, np.asarray([0] * SLACK_VAR_COUNT)]), tableau, basis, b)
        non_basic = [i for i in range(SLACK_VAR_COUNT + VAR_COUNT) if i not in basis]
        t, obj, basis, values, status = simplex_loop(tableau, basis, non_basic)
    else:
        # Directly going to second step as artificial variables are not needed (trivial basis exists)
        t, obj, basis, values, status = simplex(A, b, np.concatenate([c, np.asarray([0] * SLACK_VAR_COUNT)]), basis, b)
    x = construct_x(basis, values)
    # Store output
    write_output(OUTPUT_PATH, obj, x, status)

two_phase_simplex(A, b, c)