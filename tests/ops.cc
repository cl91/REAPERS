#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#define REAPERS_FP64
#include <reapers.h>

using namespace REAPERS;
using MatrixType = HostSumOps<>::MatrixType;
using EigenVals = HostSumOps<>::EigenVals;
using EigenVecs = HostSumOps<>::EigenVecs;

auto i = complex<>{0,1};
auto id = SpinOp<>::identity();
auto sx = SpinOp<>::sigma_x(0);
auto sy = SpinOp<>::sigma_y(0);
auto sz = SpinOp<>::sigma_z(0);
auto sx1 = SpinOp<>::sigma_x(1);
auto sy1 = SpinOp<>::sigma_y(1);
auto sz1 = SpinOp<>::sigma_z(1);
auto sx2 = SpinOp<>::sigma_x(2);
auto sy2 = SpinOp<>::sigma_y(2);
auto sz2 = SpinOp<>::sigma_z(2);
auto tol = epsilon<>()*1e2;
auto sqrt2 = std::sqrt<DefFpType>(2);

TEST_CASE("single site multiplication") {
    CHECK(id * id == id);
    CHECK(id * sx == sx);
    CHECK(id * sy == sy);
    CHECK(id * sz == sz);
    CHECK(sx * id == sx);
    CHECK(sx * sx == id);
    CHECK(sx * sy == i*sz);
    CHECK(sx * sz == -i*sy);
    CHECK(sy * id == sy);
    CHECK(sy * sx == -i*sz);
    CHECK(sy * sy == id);
    CHECK(sy * sz == i*sx);
    CHECK(sz * id == sz);
    CHECK(sz * sx == i*sy);
    CHECK(sz * sy == -i*sx);
    CHECK(sz * sz == id);
}

TEST_CASE("matrix representation") {
    MatrixType m(1,1);
    m << 1;
    SumOps ops = id;
    CHECK(ops.get_matrix(0) == m);
    m.resize(2,2);
    m << 1, 0, 0, 1;
    CHECK(ops.get_matrix(1) == m);
    ops = sx;
    m << 0, 1, 1, 0;
    CHECK(ops.get_matrix(1) == m);
    ops = sy;
    m << 0, -i, i, 0;
    CHECK(ops.get_matrix(1) == m);
    ops = sz;
    m << 1, 0, 0, -1;
    CHECK(ops.get_matrix(1) == m);
    m.resize(4,4);
    ops = id * id;
    m << 1, 0, 0, 0,
	0, 1, 0, 0,
	0, 0, 1, 0,
	0, 0, 0, 1;
    CHECK(ops.get_matrix(2) == m);
    ops = id * sx;
    m << 0, 1, 0, 0,
	1, 0, 0, 0,
	0, 0, 0, 1,
	0, 0, 1, 0;
    CHECK(ops.get_matrix(2) == m);
    ops = id * sy;
    m << 0, -i, 0, 0,
	i, 0, 0, 0,
	0, 0, 0, -i,
	0, 0, i, 0;
    CHECK(ops.get_matrix(2) == m);
    ops = id * sz;
    m << 1, 0, 0, 0,
	0, -1, 0, 0,
	0, 0, 1, 0,
	0, 0, 0, -1;
    CHECK(ops.get_matrix(2) == m);
    ops = sx1 * id;
    m << 0, 0, 1, 0,
	0, 0, 0, 1,
	1, 0, 0, 0,
	0, 1, 0, 0;
    CHECK(ops.get_matrix(2) == m);
    ops = sx1 * sx;
    m << 0, 0, 0, 1,
	0, 0, 1, 0,
	0, 1, 0, 0,
	1, 0, 0, 0;
    CHECK(ops.get_matrix(2) == m);
    ops = sx1 * sy;
    m << 0, 0, 0, -i,
	0, 0, i, 0,
	0, -i, 0, 0,
	i, 0, 0, 0;
    CHECK(ops.get_matrix(2) == m);
    ops = sx1 * sz;
    m << 0, 0, 1, 0,
	0, 0, 0, -1,
	1, 0, 0, 0,
	0, -1, 0, 0;
    CHECK(ops.get_matrix(2) == m);
    ops = sy1 * id;
    m << 0, 0, -i, 0,
	0, 0, 0, -i,
	i, 0, 0, 0,
	0, i, 0, 0;
    CHECK(ops.get_matrix(2) == m);
    ops = sy1 * sx;
    m << 0, 0, 0, -i,
	0, 0, -i, 0,
	0, i, 0, 0,
	i, 0, 0, 0;
    CHECK(ops.get_matrix(2) == m);
    ops = sy1 * sy;
    m << 0, 0, 0, -1,
	0, 0, 1, 0,
	0, 1, 0, 0,
	-1, 0, 0, 0;
    CHECK(ops.get_matrix(2) == m);
    ops = sy1 * sz;
    m << 0, 0, -i, 0,
	0, 0, 0, i,
	i, 0, 0, 0,
	0, -i, 0, 0;
    CHECK(ops.get_matrix(2) == m);
    ops = sz1 * id;
    m << 1, 0, 0, 0,
	0, 1, 0, 0,
	0, 0, -1, 0,
	0, 0, 0, -1;
    CHECK(ops.get_matrix(2) == m);
    ops = sz1 * sx;
    m << 0, 1, 0, 0,
	1, 0, 0, 0,
	0, 0, 0, -1,
	0, 0, -1, 0;
    CHECK(ops.get_matrix(2) == m);
    ops = sz1 * sy;
    m << 0, -i, 0, 0,
	i, 0, 0, 0,
	0, 0, 0, i,
	0, 0, -i, 0;
    CHECK(ops.get_matrix(2) == m);
    ops = sz1 * sz;
    m << 1, 0, 0, 0,
	0, -1, 0, 0,
	0, 0, -1, 0,
	0, 0, 0, 1;
    CHECK(ops.get_matrix(2) == m);
}

template <typename T>
bool almost_equal(const T &m0, const T &m1)
{
    return (m0 - m1).norm() < tol;
}

bool eigensys_ok(SpinOp<> op, int len, EigenVals evals, EigenVecs evecs)
{
    SumOps ops(op);
    auto eisys = ops.get_eigensystem(len);
    auto vals = std::get<0>(eisys);
    auto vecs = std::get<1>(eisys);
    return almost_equal(evals, vals) && almost_equal(evecs, vecs);
}

TEST_CASE("eigenvalues and eigenvectors") {
    EigenVals vals(2);
    EigenVecs vecs(2, 2);
    vals << 1, 1;
    vecs << 1, 0,
	0, 1;
    CHECK(eigensys_ok(id, 1, vals, vecs));
    vals << -1, 1;
    vecs << -1.0/sqrt2, 1.0/sqrt2,
	1.0/sqrt2, 1.0/sqrt2;
    CHECK(eigensys_ok(sx, 1, vals, vecs));
    vecs << 1.0/sqrt2, 1.0/sqrt2,
	-i/sqrt2, i/sqrt2;
    CHECK(eigensys_ok(sy, 1, vals, vecs));
    vecs << 0, 1,
	1, 0;
    CHECK(eigensys_ok(sz, 1, vals, vecs));
}

TEST_CASE("matrix exponentials") {
    SumOps ops(id);
    EigenVecs m(2, 2);
    m << 2.71828182845905, 0,
	0, 2.71828182845905;
    CHECK(almost_equal(ops.matexp(1, 1), m));
    ops = sx;
    m << 1.54308063481524, 1.17520119364380,
	1.17520119364380, 1.54308063481524;
    CHECK(almost_equal(ops.matexp(1, 1), m));
    ops = sy;
    m << 1.54308063481524, -1.17520119364380*i,
	1.17520119364380*i, 1.54308063481524;
    CHECK(almost_equal(ops.matexp(1, 1), m));
    ops = sz;
    m << 2.71828182845905, 0,
	0, 0.367879441171442;
    CHECK(almost_equal(ops.matexp(1, 1), m));
    ops = sx;
    m << 0.540302305868140, 0.841470984807897*i,
	0.841470984807897*i, 0.540302305868140;
    CHECK(almost_equal(ops.matexp(i, 1), m));
    ops = sy;
    m << 0.540302305868140, 0.841470984807897,
	-0.841470984807897, 0.540302305868140;
    CHECK(almost_equal(ops.matexp(i, 1), m));
    ops = sz;
    m << 0.540302305868140 + 0.841470984807897*i, 0,
	0, 0.540302305868140 - 0.841470984807897*i;
    CHECK(almost_equal(ops.matexp(i, 1), m));
}
