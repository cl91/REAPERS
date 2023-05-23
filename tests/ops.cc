#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#define REAPERS_FP64
#include <reapers.h>

using namespace REAPERS;
using namespace REAPERS::Model;
using MatrixType = HostSumOps<>::MatrixType;
using EigenVals = HostSumOps<>::EigenVals;
using EigenVecs = HostSumOps<>::EigenVecs;

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
double sqrt2 = std::sqrt(2.0);

TEST_CASE("single site multiplication") {
    auto i = complex<>{0,1};
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
    auto i = complex<>{0,1};
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
    auto i = complex<>{0,1};
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
    auto i = complex<>{0,1};
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

TEST_CASE("spin operator equality") {
    CHECK(id == id);
    CHECK(id != sx);
    CHECK(id != sy);
    CHECK(id != sz);
    CHECK(sx != id);
    CHECK(sx == sx);
    CHECK(sx != sy);
    CHECK(sx != sz);
    CHECK(sy != id);
    CHECK(sy != sx);
    CHECK(sy == sy);
    CHECK(sy != sz);
    CHECK(sz != id);
    CHECK(sz != sx);
    CHECK(sz != sy);
    CHECK(sz == sz);
    CHECK(sx1 != sx);
    CHECK(sx2 != sx);
    CHECK(sx1 == sx1);
    CHECK(sx2 == sx2);
    CHECK(sy1 != sy);
    CHECK(sy2 != sy);
    CHECK(sy1 == sy1);
    CHECK(sy2 == sy2);
    CHECK(sz1 != sz);
    CHECK(sz2 != sz);
    CHECK(sz1 == sz1);
    CHECK(sz2 == sz2);
}

TEST_CASE("commutativity of addition and multiplication") {
    CHECK(id + sx == sx + id);
    CHECK(id + sy == sy + id);
    CHECK(id + sz == sz + id);
    CHECK(sx + sy == sy + sx);
    CHECK(sx + sz == sz + sx);
    CHECK(sz + sy == sy + sz);
    CHECK(sx1 + sy == sy + sx1);
    CHECK(sx1 + sz == sz + sx1);
    CHECK(sz1 + sy == sy + sz1);
    CHECK(sx2 + sy == sy + sx2);
    CHECK(sx2 + sz == sz + sx2);
    CHECK(sz2 + sy == sy + sz2);
    CHECK(sx2 + sy1 == sy1 + sx2);
    CHECK(sx2 + sz1 == sz1 + sx2);
    CHECK(sz2 + sy1 == sy1 + sz2);
}

TEST_CASE("subtraction") {
    CHECK((id - id == 0));
    CHECK((sx - sx == 0));
    CHECK((sy - sy == 0));
    CHECK((sz - sz == 0));
    CHECK((sx1 - sx1 == 0));
    CHECK((sy1 - sy1 == 0));
    CHECK((sz1 - sz1 == 0));
    CHECK((sx2 - sx2 == 0));
    CHECK((sy2 - sy2 == 0));
    CHECK((sz2 - sz2 == 0));
    CHECK((sx2 - sx != 0));
    CHECK((sy2 - sy != 0));
    CHECK((sz2 - sz != 0));
    CHECK((sx2 - sx1 != 0));
    CHECK((sy2 - sy1 != 0));
    CHECK((sz2 - sz1 != 0));
}

TEST_CASE("fermion ops commutative relation") {
    double one = (1.0/sqrt2)*(1.0/sqrt2)*2;
    for (int N = 4; N <= 20; N += 2) {
	SYK syk(N, 0.0, 1.0, true);
	auto gamma = syk.fermion_ops();
	SYK syk2(N, 0.0, 1.0, false);
	auto gamma2 = syk2.fermion_ops();
	for (int i = 0; i < N; i++) {
	    for (int j = 0; j < N; j++) {
		auto comm = gamma[i]*gamma[j]+gamma[j]*gamma[i];
		if (i != j) {
		    CHECK(comm == 0);
		} else {
		    // Due to numerical limit 2*(1/sqrt2)^2 is not exactly 1.
		    CHECK(comm == one);
		}
		auto comm2 = gamma2[i]*gamma2[j]+gamma2[j]*gamma2[i];
		CHECK((comm2 == ((i == j) ? 2 : 0)));
	    }
	}
    }
}
