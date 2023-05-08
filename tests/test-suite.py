#!/usr/bin/python
#
# Test suite of the REAPERS library. We basically compare the results from
# our library with those from numpy and dynamite.

import os, subprocess, json, argparse
import numpy as np
import dynamite.computations as cp
from dynamite.states import State
from dynamite.operators import identity, sigmax, sigmay, sigmaz, op_sum

# Default tolerance of the difference between two vectors/matrices
def_tol = 1e-10

parser = argparse.ArgumentParser(prog = 'test-suite.py',
                                 description = 'Test suite for the REAPERS library '
                                 'by comparing with numpy and dynamite',
                                 formatter_class = argparse.RawDescriptionHelpFormatter)
parser.add_argument('-v', '--verbose', action = 'store_true',
                    help = "Print diagnostics and other runtime information.")
parser.add_argument('-n', '--no-rebuild', action = 'store_true',
                    help = "Don't rebuild the testing service program.")
parser.add_argument('-c', '--compile-only', action = 'store_true',
                    help = "Build the testing service program but don't run any tests.")
parser.add_argument('--gpu', action = 'store_true',
                    help = "Run (or build) the GPU version of the test program.")
parser.add_argument('--intel', action = 'store_true',
                    help = "Run (or build) the Intel CPU version of the test program.")
parser.add_argument('--tol', type = float, default = def_tol,
                    help = 'Specifies the inverse temperature of the simulation.')
parser.add_argument('--run-test', nargs='+', default=[],
                    help = 'Run the specified test.')
parser.add_argument('--run-funclet', nargs='+', default=[],
                    help = 'Run the specified funclet.')
cfg = parser.parse_args()

if cfg.gpu:
    CC="nvcc"
elif cfg.intel:
    CC="icpx"
else:
    CC="c++"

CFLAGS=['-std=c++20', '-Wall', '-Wno-maybe-uninitialized',
        '-Wno-uninitialized', '-I/usr/include/eigen3',
        '-I../inc', '-lboost_program_options']

if not cfg.intel:
    CFLAGS.append('-fopenmp')
if not cfg.gpu:
    CFLAGS.append('-DREAPERS_NOGPU')
if cfg.intel or cfg.gpu:
    CFLAGS.extend(['-DNDEBUG', '-O3', '-ffast-math', '-mtune=native'])
else:
    CFLAGS.extend(['-g', '-DREAPERS_DEBUG'])

if cfg.gpu:
    CFLAGS.extend(['-forward-unknown-to-host-compiler', '-march=native',
                   '-Wno-unknown-pragmas', '-lcublas', '-lcusolver', '-lcurand',
                   '../src/krnl.cu'])

if cfg.intel:
    CFLAGS.extend(['-xhost', '-fiopenmp', '-Wno-tautological-constant-compare',
                   '-Wno-unused-but-set-variable'])

SERVICE_SRC="service.cc"

if cfg.gpu:
    SERVICE_EXE="service-gpu"
elif cfg.intel:
    SERVICE_EXE="service-intel"
else:
    SERVICE_EXE="service"

# Switch current working directory to where the script is located
os.chdir(os.path.dirname(__file__))

# Build the service file
if not cfg.no_rebuild:
    subprocess.run([CC] + CFLAGS + [SERVICE_SRC, '-o', SERVICE_EXE], check=True)

if cfg.compile_only:
    exit(0)

# Run the service executable
def run_exec(reqjson):
    res = subprocess.run(['./'+SERVICE_EXE], input = reqjson, encoding='utf-8',
                         capture_output=True)
    if res.returncode != 0 or cfg.verbose:
        print(f"Testing service executable returned code {res.returncode}")
        print("Request was:")
        print(reqjson)
        print("Standard output:")
        print(res.stdout)
        print("Standard error:")
        print(res.stderr)
        if res.returncode != 0:
            exit(res.returncode)
    return res.stdout

# Convert the response SpinOp object to a dynamite spin op
def to_dynamite_op(spin_op):
    bits = spin_op["bits"]
    coeff = spin_op["coeff"]
    op = identity()
    site = 0
    while bits != 0:
        sop = bits & 3
        if sop == 0:
            pass
        elif sop == 1:
            op = sigmax(site) * op
        elif sop == 2:
            op = sigmaz(site) * op
        else:
            op = sigmay(site) * op
        site += 1
        bits >>= 2
    op *= coeff
    return op

# Convert the list of dynamite spin objects to dynamite opsum
def to_dynamite_opsum(oplist, spinlen):
    res = op_sum([to_dynamite_op(op) for op in oplist])
    res.L = spinlen
    return res

# Convert the petsc sparse matrix to a numpy dense matrix
def pestc_mat_to_numpy(mat, spinlen):
    dim = 1 << spinlen
    res = np.zeros((dim,dim), dtype=complex)
    for rowidx in range(dim):
        (colidxs, vals) = mat.getRow(rowidx)
        for i in range(len(colidxs)):
            res[rowidx, colidxs[i]] = vals[i]
    return res

# Returns the Frobenius norm of the differences of two matrices/vectors
def diff_norm(mat0, mat1):
    return np.linalg.norm(mat0 - mat1)

# Returns true if both numpy matrices are equal up to the
# specified tolerance.
def almost_equal(mat0, mat1):
    nrm = diff_norm(mat0, mat1)
    if nrm < cfg.tol:
        return True
    else:
        print(f"nrm = {nrm}\nmat0 = {mat0}\nmat1 = {mat1}")
        return False

# Convert the complex vector in the response object to numpy array.
def rvec_to_numpy(rvec):
    return np.array(rvec)

# Convert the complex vector in the response object to numpy array.
def cvec_to_numpy(cvec):
    arr = []
    for c in cvec:
        arr.append(c[0] + 1j * c[1])
    return np.array(arr)

# Convert the complex matrix in the response object to numpy ndarray.
# cmat is assumed to be in row-major order.
def cmat_to_numpy(cmat):
    dim = (int)(np.sqrt(len(cmat)))
    assert(dim*dim == len(cmat))
    res = np.zeros((dim,dim), dtype=complex)
    for row in range(dim):
        for col in range(dim):
            c = cmat[row*dim+col]
            res[row,col] = c[0] + 1j * c[1]
    return res

# Convert the numpy array to a dynamite State
def to_dynamite_state(s):
    st = State((int)(s["len"]))
    st.vec.array = cvec_to_numpy(s["vec"])
    st.set_initialized()
    return st

# Define the tests we are going to run
def gen_ham_genreq(N):
    return {"request" : "gen-ham", "params" : { "N" : N,
                                                "sparsity" : 0.0,
                                                "regularize" : False }}

def gen_ham_test(req, reply):
    spinlen = (int)(reply["N"]/2)
    dyn_ops = to_dynamite_opsum(reply["ham"], spinlen)
    dyn_mat = pestc_mat_to_numpy(dyn_ops.get_mat(), spinlen)
    mat = cmat_to_numpy(reply["mat"])
    return almost_equal(dyn_mat, mat)

def get_eigensys_genreq(N):
    return {"request" : "get-eigensys", "params" : { "N" : N,
                                                     "sparsity" : 0.0,
                                                     "regularize" : False }}

def get_eigensys_test(req, reply):
    spinlen = (int)(reply["N"]/2)
    dyn_ops = to_dynamite_opsum(reply["ham"], spinlen)
    dyn_mat = pestc_mat_to_numpy(dyn_ops.get_mat(), spinlen)
    (evals_numpy, evecs_numpy) = np.linalg.eigh(dyn_mat)
    evals = rvec_to_numpy(reply["eigenvals"])
    # In general each eigenvector can differ by a phase factor so we don't
    # bother comparing the eigenvectors, but we do check that the eigen-
    # vectors returned by the service program satisfies the eigen problem.
    evecs = cmat_to_numpy(reply["eigenvecs"])
    if cfg.verbose:
        print(f"evals from REAPERS: {evals}\nevals from numpy: {evals_numpy}")
        print(f"evecs from REAPERS: {evecs}\nevecs from numpy: {evecs_numpy}")
    if not almost_equal(evals, evals_numpy):
        if cfg.verbose:
            printf("Eigenvalues do not match")
        return False
    return almost_equal(dyn_mat.dot(evecs), evecs.dot(np.diag(evals)))

def evolve_state_genreq(N, t, beta, exdiag = False):
    return {"request" : "evolve-state", "params" : { "N" : N,
                                                     "sparsity" : 0.0,
                                                     "regularize" : False,
                                                     "t" : t,
                                                     "beta" : beta,
                                                     "exdiag" : exdiag }}

def evolve_state_test(req, reply):
    spinlen = (int)(reply["N"]/2)
    t = req["t"]
    beta = req["beta"]
    dyn_ops = to_dynamite_opsum(reply["ham"], spinlen)
    init_state = cvec_to_numpy(reply["init-state"]["vec"])
    final_state = cvec_to_numpy(reply["final-state"]["vec"])
    if req["exdiag"]:
        dyn_mat = pestc_mat_to_numpy(dyn_ops.get_mat(), spinlen)
        (evals, evecs) = np.linalg.eigh(dyn_mat)
        expevals = np.exp((-beta-1j*t)*evals)
        expmat = evecs.dot(np.diag(expevals)).dot(np.conj(np.transpose(evecs)))
        exdiag_final_state = np.matmul(expmat, init_state)
        return almost_equal(final_state, exdiag_final_state)
    else:
        dyn_init_state = to_dynamite_state(reply["init-state"])
        dyn_final_state = dyn_ops.evolve(dyn_init_state, t-1j*beta, algo='krylov')
        return almost_equal(final_state, dyn_final_state.vec.getArray())

def evolve_state_printnorm(req, reply):
    spinlen = (int)(reply["N"]/2)
    t = req["t"]
    beta = req["beta"]
    dyn_ops = to_dynamite_opsum(reply["ham"], spinlen)
    init_state = cvec_to_numpy(reply["init-state"]["vec"])
    final_state = cvec_to_numpy(reply["final-state"]["vec"])
    dyn_init_state = to_dynamite_state(reply["init-state"])
    dyn_final_state = dyn_ops.evolve(dyn_init_state, t-1j*beta, algo='krylov')
    res = diff_norm(final_state, dyn_final_state.vec.getArray())
    print(f"{reply['N']} {t} {beta} {res}")

tests = { "gen-ham" : (gen_ham_genreq, gen_ham_test),
          "get-eigensys" : (get_eigensys_genreq, get_eigensys_test),
          "evolve-state" : (evolve_state_genreq, evolve_state_test) }

funclets = { "evolve-state" : (evolve_state_genreq, evolve_state_printnorm) }

# Get the test or funclet to run
def get(name, d, ty_str):
    if not name in d:
        print(f"Specified {ty_str} {name} does not exist.")
        exit(1)
    return d[name]

# Generate and submit the request and parse the service response
def run(req_gen, cfg_param_list):
    params = [eval(e) for e in cfg_param_list[1:]]
    req = req_gen(*params)
    reqjson = json.dumps(req)
    reply = json.loads(run_exec(reqjson))
    return (req, reply)

if cfg.run_test:
    (req_gen, test) = get(cfg.run_test[0], tests, "test")
    (req, reply) = run(req_gen, cfg.run_test)
    if not test(req["params"], reply):
        print(f"Test \'{' '.join(cfg.run_test)}\' failed.")
        exit(1)
    else:
        print(f"Test \'{' '.join(cfg.run_test)}\' succeeded.")

if cfg.run_funclet:
    (req_gen, funclet) = get(cfg.run_funclet[0], funclets, "funclet")
    (req, reply) = run(req_gen, cfg.run_funclet)
    funclet(req["params"], reply)
