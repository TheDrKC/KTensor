# KTensor

KTensor is a header-only library enabling natural Einstein summation over
arbitrary array-like structures in modern C++. Its motivating philosophy is that manually-written loops create opportunities for bugs while obscuring the code's mathematical intent and thus should be avoided whenever
possible. Accordingly, KTensor targets four goals:

1. Minimize the number of hand-written loops by applying implicit summation
2. Simplify the translation of pen-and-paper formulas to C++ by mimicking natural mathematical notation
3. Ensure correctness of loop extents by inferring them from array extents
4. Detect bugs early by not allowing invalid expressions to compile

Though care has been taken not to degrade runtime performance in pursuing
these goals, deliberately seeking high performance is left for future
development. So KTensor currently makes no effort to exploit SIMD instructions,
apply mathematical simplifications at compile time, target GPUs, or anything
similar. What it does do is transform concise tensor expressions into equivalent
nested loops, which any modern compiler should then translate into assembly
code similar or identical to that which would have been produced if you had
(correctly) written those loops yourself. So expect similar performance.

## Einstein Summation

KTensor implements a more general version of Einstein summation than what one
typically encounters in physics. The <a href="https://mathworld.wolfram.com/EinsteinSummation.html">usual physics version</a>
has three rules:

>  Repeated indices are implicitly summed over.
>
>  Each index can appear at most twice in any term.
>
>  Each term must contain identical non-repeated indices. 

The second requirement ensures that expressions will be proper tensors, i.e.
will transform as tensors under changes of basis. As writers of software we
are not concerned with that property but often need to multiply three arrays
componentwise or add the nth element of a vector to each entry in the nth row
of a matrix, so KTensor drops the second and third rules. The KTensor version
of Einstein summation is described by a single rule:

> 1. Repeated indices within an additive term imply summation over all values of those indices

"Additive terms" are terms separated by the operators `+` or `-`. The following
are all examples of valid KTensor expressions with two additive terms:

~~~c++
A(i,j) + B(k,k)
A(i,k)*B(k,j) - C(i,k)*D(k,j)
i*(j + k)
(cos(i)-j)*B(i,4,k,k)
~~~

Dropping the last two rules permits expressions that could be ambiguous but
which are nonetheless useful to a programmer. For example, to extract the
diagonal of a matrix into a vector one might wish to write:

~~~c++
v(k) = A(k,k);
~~~

which could also be interpreted as setting each entry of $v$ to the trace of $A$.
Of course, if one wanted to do that it could be unmistakeably communicated by
simply using a different index variable on the right-hand side:

~~~c++
v(k) = A(p,p);
~~~

So in order to be maximally useful, KTensor interprets the first form as
extracting the diagonal and the second as assigning the trace. KTensor
achieves this behavior by applying the following rules:

> 2. Implicit summation occurs only at assignment.
>
> 3. Each index on the left-hand side (LHS) of an assignment implies a loop over the possible values of that index.
>
> 4. Implicit summation according to rule 1 occurs for indices appearing on the right-hand side (RHS) which do not appear on the LHS.

Rule 3 gives rise to a set of nested loops which cover all entries of the
LHS tensor. Inside all those loops are other loops which perform the
implicit summations.

## Tensor Objects

As a data structure, a tensor need only be akin to a multidimensional array.
KTensor implements this idea through the `MDTensor` template class which
extends <a href="https://en.cppreference.com/w/cpp/container/mdspan">`mdspan`</a>
and takes the same template parameters and constructor arguments. KTensor
therefore provides Einstein summation capability to any array-like object for
which a `mdspan` can be formed. For simplicity of typing, one can
alternatively use the alias `KTensor::Tensor` which is equivalent to a
`MDTensor` with default layout and accessor policies,
and for which the extent parameters can be given directly instead of in a
`std::extents` type. For example, in the following snippet `M1`, `M2`, and `M3`
are all of the same type but declared differently:

~~~c++
#include "KTensor/KTensor.hpp"

using namespace KTensor;

int main(){
  std::vector<double> x;
  int extent2;
  std::cin >> extent2;
  x.resize(3*extent2);
  // Using std here assumes C++23
  // See the "Getting Started" section for details
  constexpr auto N = std::dynamic_extent;

  KTensor::MDTensor<double,std::extents<std::size_t,3,N>, std::layout_right, std::default_accessor<double>> M1(x.data(), extent2);
  KTensor::MDTensor<double,std::extents<std::size_t,3,N>> M2(x.data(), extent2);
  KTensor::Tensor<double,3,N> M3(x.data(), extent2);
}
~~~

MDTensors can be of arbitrary rank but that rank must be known at compile time.

## Indices and Tensor Expressions

An *index* is a placeholder symbol indicating a variable taking consecutive
integer values starting from zero. Indices must be created before use and are
associated with a character given as a template parameter:

~~~c++
KTensor::Index<'i'> i;     // Name matches character - this is good practice
KTensor::Index<'j'> j;     // Name matches character - this is good practice
KTensor::Index<'q'> bad_idea;  // Name does not match character - this is atrociously bad practice, but permitted
~~~
KTensor parses expressions *based on the template parameter character*, not the
name of the `Index` object. So for the clearest code one should **always** make
the object name match the template parameter.

Indices are mainly used as arguments to MDTensor objects, thereby forming
tensor expressions:

~~~c++
// i and j are Index objects
// A and A_transpose are Tensors (instances of the MDTensor class)
A_tranpose(i,j) = A(j,i);  // A_transpose(i,j) and A(j,i) are tensor expressions
~~~

They are also tensor expressions in their own right, however, so can also be
used to initialize MDTensor objects:

~~~c++
A(i,j) = i + j;
~~~
This expression is equivalent to the hand-written assignment:
~~~c++
for(int ix=0; ix<i_extent; ++ix){
  for(int jx=0; jx<j_extent; ++jx){
    A(ix,jx) = ix + jx;
  }
}
~~~
where `i_extent` and `j_extent` are respectively the extents of the first and
second subscripts of `A`. Since the extents of `A` were set when it was
declared, KTensor can infer these extents (even those set dynamically) and sets
the loop extents to match them. Observe the vastly greater simplicity of the
KTensor expression that results.

Since an index can take any nonnegative integer value, they cannot be used
together in a contraction but can be used in a contraction with a tensor
expression with definite extents:

~~~c++
Index<'i'> i;
Index<'j'> j;
Index<'k'> k;
std::array<int,9> A_array;
std::array<int,3> v_array;
KTensor::Tensor<int,3,3> A(A_array.data());
KTensor::Tensor<int,3> V(v_array.data());
//A(i,j) = i + j - k*k;  // Compile-time error: k*k would imply summation over all nonnegative integers, an infinite loop
A(i,j) = i + j - k*V(k);  // Valid since V has a defined extent
~~~

Tensor expressions can be formed with mixed integer and index subscripts:

~~~c++
A(2,j) = j;
A(i,3) = V(i);
~~~

This feature can be used to form loops over only part of a Tensor if desired:

~~~c++
std::array<int,16> A_array;
std::array<int,4> x_array, b_array;
KTensor::Tensor<int,4,4> A(A_array.data());
KTensor::Tensor<int,4> x(x_array.data());
KTensor::Tensor<int,4> b(b_array.data());
KTensor::Index<'j'> j;

// Initialize A and x

// Set only the first two elements of b = A*x
for(int ii = 0; ii<2; ++ii){
  // Implicit summation occurs over j
  b(ii) = A(ii,j)*x(j);
}
~~~

Tensor expressions which evaluate to a scalar can be assigned to a scalar using
an explicit cast. Note that the *entire* expression must be in parentheses
otherwise the cast will apply only to the first term in the expression giving a
compilation error.

~~~c++
double frobenius_norm2 = (double) (A(i,j)*A(j,i));
~~~

## Functions

KTensor supports arbitrary functions of tensor expressions, with many standard
mathematical functions already available. For example:

~~~c++
A(i,j) = cos(i-j) + expm1(k*V(k) - j)*atan2(k,j);
~~~

One can also include one's own functions in KTensor expressions using the
`make_function` function:

~~~c++
double my_function(double x, double y){
  // ...
}
// Passes the result of V(i)+j as the first argument to my_function,
// and the result of exp(j) as the second.
A(i,j) = KTensor::make_function(my_function, V(i)+j, exp(j));
~~~

## Inference of Loop Extents

The extent associated with an index is *automatically inferred* from the
expression in which it appears. For instance, if `A` is declared with extents
of 3 for the first subscript and 7 for the second, then for the assignment:

~~~c++
A(i,j) = B(j,k)*C(k,i);
~~~

to compile `B` must have been declared with extent 7 for its first subscript,
and `C` with extent 3 for its second. The second subscript of `B` and first
subscript of `C` can have any extent but must both be the same.

KTensor supports dynamic extents (i.e. extents which are not known until run
time), but of course cannot enforce consistency between dynamic extents at
compile time since they are not known. So at compile time, KTensor regards all
dynamic extents as equal to each other and not equal to any static extent. This
means that an expression is invalid if both a dynamic and a static extent are
inferred for the same index. For example:

~~~c++
constexpr auto N = std::dynamic_extent;
int d;
std::cin >> d;  // A user-selected extent
KTensor::Tensor<double,3,N> A(ptr_to_data1,d);
KTensor::Tensor<double,N,4> B(ptr_to_data2,d);
KTensor::Tensor<double,3,4> X(ptr_to_data3);
KTensor::Tensor<double,3,4> C(ptr_to_data3);
KTensor::Index<'d'> d;
KTensor::Index<'i'> i;
KTensor::Index<'j'> j;

// Implicit summation will occur over the index d
X(i,j) = A(i,d)*B(d,j);  // Valid - inferred extent for d is dynamic in both A and B
X(i,j) = A(i,d)*C(d,j);  // Not valid - inferred extent for d is dynamic in A but static (3) in C
~~~

KTensor does **not** check at runtime for matching dynamic extents.

## Special Tensors

KTensor currently supports two special tensors:

- `KroneckerDelta<S>` for the <a href="https://mathworld.wolfram.com/KroneckerDelta.html">Kronecker delta</a> with return type `S`:

$$ \delta_{ij} = \begin{cases}1 & i = j \\ 0 & i \neq j \end{cases} $$

~~~c++
KTensor::KroneckerDelta<double> delta;
A(i,j) = delta(i,j);  // Initialize A to the identity matrix
~~~

- `LeviCivita<S>` for the three-index <a href="https://mathworld.wolfram.com/PermutationSymbol.html">Levi-Civita symbol</a> with return type `S`:

$$ \epsilon_{ijk} = \begin{cases} 1 & \text{ijk is an even permutation} \\ -1 & \text{ijk is an odd permutation} \\ 0 & \text{otherwise} \end{cases} $$

~~~c++
KTensor::LeviCivita<double> epsilon;
uxv(i) = epsilon(i,j,k)*u(j)*v(k);  // Cross product of vectors u and v
~~~

## Compile-Time Checks

Valid KTensor expressions satisfy the following rules, to some of which we have
already alluded:

1. Each index on the RHS must appear on the LHS unless it is implicitly summed over
2. The extent of an index on the LHS must match its extents everywhere it appears on the RHS
3. If an index is involved in implicit summation, it must have the same extent whereever it is summed over
4. Implicit summation occurs only over indices with finite extents
5. Only expressions that reduce to a scalar can be assigned to scalar types variables (for which one must use an explicit cast)

Violations of these rules result in compilation errors, however these errors
tend to be very verbose even for small problems because KTensor expressions tend
to involve multiple levels of nested template types. So to aid debugging,
KTensor includes static assertions that print more concise messages about the
nature of the problem in all caps - e.g., "AN INDEX ON THE RHS IS NOT ON THE LHS
 AND IS NOT IMPLICITLY SUMMED OVER". Look for these messages in the compiler
 output instead of trying to decipher the nested templates.

# Getting Started

KTensor uses <a href="https://cmake.org/">CMake</a> (version 3.20 or higher)
for configuration and installation. A compiler which supports at least C++20 is
required as KTensor makes heavy use of C++20 language features.

1. After downloading the code, create a build directory:

~~~bash
mkdir build
cd build
ccmake /path/to/KTensor/
~~~

2. Set `CMAKE_INSTALL_PREFIX` to the path where you want the KTensor headers to
be installed.

3. KTensor uses `mdspan`, which is available as `std::mdspan` in C++23 but not
in C++20. If building with C++20, one must first obtain and install the
<a href="https://github.com/kokkos/mdspan">Kokkos version of `mdspan`</a>. Then,
 set `KTENSOR_MDSPAN_NAMESPACE` to `Kokkos` and `KTENSOR_KOKKOS_MDSPAN_DIR`
to the path to the directory containing the installed Kokkos version of
`mdspan.hpp`. If using C++23 instead, simply set `KTENSOR_MDSPAN_NAMESPACE` to
`std`.

4. Configure and generate the build files.
5. Build and install KTensor. From the build directory, do:

~~~bash
cmake --build .
cmake --install .
~~~

6. (Optional) If `ENABLE_TESTS` was on, the tests may now be built and run with
CTest. The test suite contains numerous tests and may take several minutes to
complete. CMake callls a Python3 script to generate the test files.

7. KTensor can now be used either as a CMake package or by adding `$CMAKE_INSTALL_PREFIX/include`
to your include paths. In your source files, include the KTensor headers like so:

~~~c++
#include "KTensor/KTensor.hpp"
#include "KTensor/IO.hpp"
~~~
The `IO.hpp` header allows MDTensors to be inserted into streams using `operator<<`.
