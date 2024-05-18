#ifndef KTENSOR_FUNCTIONS_HPP
#define KTENSOR_FUNCTIONS_HPP

#include <cmath>

namespace KTensor{

/** Helper structs to define the return type of scalars and TensorExpressions
 * @tparam T Any type
*/
template<typename T>
struct ExprReturn
{
  using type = T; // Scalar return type is its own type
  using tensorexpr_t = ScalarExpr<T>;
  constexpr static int_t rank = 0;
};

/** Partial specialization for TensorExpression types, for which
 * the return type is already a member type.
*/
template<TensorExpr_c T>
struct ExprReturn<T>
{
  using type = T::return_type;
  using tensorexpr_t = T;
  constexpr static int_t rank = T::map_t::rank;
};

/** Representation of an arbitrary function of an arbitrary
 * mix of scalar and TensorExpression arguments.
 * @tparam F Type of the function
 * @tparam Args_t... Types of the arguments
*/
template<typename F, typename...Args_t>
class FunctionOfExprs{
  public:
    using map_t = typename JoinMapsFromTypes<Args_t...>::type;
    using return_type = std::invoke_result_t<F, typename ExprReturn<Args_t>::type...>;
  private:
    F function;
    using tuple_t = std::tuple<typename ExprReturn<Args_t>::tensorexpr_t...>;
    tuple_t args;
    constexpr static int_t nargs = sizeof...(Args_t);

    consteval static auto setup(){
      std::array<int_t,nargs> const argranks = {{ExprReturn<Args_t>::rank...}};
      std::array<int_t,1+nargs> cumulative_argranks;
      cumulative_argranks[0] = 0;
      for(int_t i=0; i<nargs; ++i){
        cumulative_argranks[i+1] = cumulative_argranks[i] + argranks[i];
      }
      return cumulative_argranks;
    }

    constexpr static auto cumulative_argranks = setup();

    /** Evaluate a single one of the arguments to the function
     * @tparam idx Position of the argument to be evaluated
     * @arg Is... Values corresponding to the index characters of map_t
     * @return Value of the argument
     */
    template<int idx>
    return_type eval_single_arg(const Multi<int_t> auto...Is) const{
      using arg_t = typename std::tuple_element_t<idx,tuple_t>;
      const std::array<int_t,sizeof...(Is)> passed_args = {{Is...}};
      return [this,&passed_args]<char...c>(chars<c...>){
        return std::get<idx>(args).subscript(passed_args[map_t::char_location_in_map(c)]...);
      }(typename arg_t::map_t::index_chars_t());
    }

  public:
    /** Constructor */
    constexpr FunctionOfExprs(F const& f_, Args_t const&... args_) : function(f_), args(args_...) {}

    /** Forward a request for the dynamic extent to the wrapped object
     * @arg n Index for which to get the extent.
    */
    int_t extent(int_t const& n) const{
      return [this,&n]<int_t...idx>(ints<idx...>){
        int_t ext = 0;
        (((n < cumulative_argranks[1+idx]) ? (ext = std::get<idx>(args).extent(n - cumulative_argranks[idx]),false) : true) && ...);
        return ext;
      }(intrange<nargs>());
    }

    consteval static bool contracting(char const i){
      return [&i]<int_t...idx>(ints<idx...>) consteval {
        return (std::tuple_element_t<idx,tuple_t>::contracting(i) || ...);
      }(intrange<nargs>());
    }

    /** Evaluate the function for a given set of specified index values
     * @arg Is... Values for the indices in map_t
     * @return Value of the function
     */
    return_type subscript(const Multi<int_t> auto...Is) const{
      return [this,&Is...]<int_t...idx>(ints<idx...>){
        return function(eval_single_arg<idx>(Is...)...);
      }(intrange<nargs>());
    }

    /** Evaluate implicit summations of arguments to the function
     * @tparam P... Free index characters
     * @arg Ps... Values corresponding to the indices P
     * @return Result of the implicit summation
     */
    template<char...P>
    return_type implicit_summation(chars<P...>, const Multi<int_t> auto...Ps) const
    requires(Addable<return_type>)
    {
      return [this, &Ps...]<int_t...idx>(ints<idx...>){
        return function(std::get<idx>(args).implicit_summation(chars<P...>(), Ps...)...);
      }(intrange<nargs>());
    }
};

/** Produce a TensorExpression wrapping an arbitrary function of arbitrary
 * scalars and/or TensorExpressions.
 * @tparam F Type of the function-like object being wrapped
 * @tparam Args_t... Types of the arguments to the function
 * @arg f The function being wrapped
 * @arg args The expressions which are arguments to the function, of which there must be at least one.
 * @return TensorExpression describing the function evaluation
 */
template<typename F, typename...Args_t>
auto make_function(F f, Args_t const&...args)
requires((sizeof...(Args_t) > 0) && (false || ... || TensorExpr_c<Args_t>))
{
  using Result_t = FunctionOfExprs<F, Args_t...>;
  using map_t = typename Result_t::map_t;
  return [&f,&args...]<char...idx>(chars<idx...>){
    return TensorExpression<Result_t, map_t, idx...>(Result_t(f, args...));
  }(typename map_t::index_chars_t());
}

/** Define an overload for a function taking a specified number of arguments
 * and returning a double
 * @arg NAME Name of the function to overload
 * @arg NARGS Number of arguments to the overloaded function
*/
#define KFUNCTION(NAME,NARGS) \
template<typename...Ts> \
requires((sizeof...(Ts) == NARGS) && \
         (false || ... || TensorExpr_c<Ts>)) \
auto NAME(Ts...X){ \
  return make_function(static_cast<double(*)(typename ExprReturn<Ts>::type...)>(std::NAME), X...); \
}

// Overload mathematical functions from the cmath header
KFUNCTION(abs,1)
KFUNCTION(exp,1)
KFUNCTION(exp2,1)
KFUNCTION(expm1,1)
KFUNCTION(log,1)
KFUNCTION(log10,1)
KFUNCTION(log2,1)
KFUNCTION(log1p,1)
KFUNCTION(pow,2)
KFUNCTION(sqrt,1)
KFUNCTION(cbrt,1)
KFUNCTION(sin,1)
KFUNCTION(cos,1)
KFUNCTION(tan,1)
KFUNCTION(asin,1)
KFUNCTION(acos,1)
KFUNCTION(atan,1)
KFUNCTION(atan2,2)
KFUNCTION(sinh,1)
KFUNCTION(cosh,1)
KFUNCTION(tanh,1)
KFUNCTION(asinh,1)
KFUNCTION(acosh,1)
KFUNCTION(atanh,1)
KFUNCTION(erf,1)
KFUNCTION(erfc,1)
KFUNCTION(tgamma,1)
KFUNCTION(lgamma,1)
KFUNCTION(beta,2)
KFUNCTION(riemann_zeta,1)
KFUNCTION(comp_ellint_1,1)
KFUNCTION(comp_ellint_2,1)
KFUNCTION(comp_ellint_3,2)
KFUNCTION(ellint_1,2)
KFUNCTION(ellint_2,2)
KFUNCTION(ellint_3,3)
KFUNCTION(expint,1)
KFUNCTION(cyl_bessel_i,2)
KFUNCTION(cyl_bessel_j,2)
KFUNCTION(cyl_bessel_k,2)
KFUNCTION(cyl_neumann,2)
KFUNCTION(sph_bessel,2)
KFUNCTION(sph_legendre,3)
KFUNCTION(sph_neumann,2)
KFUNCTION(laguerre,2)
KFUNCTION(assoc_laguerre,3)
KFUNCTION(legendre,2)
KFUNCTION(assoc_legendre,3)
KFUNCTION(hermite,2)

} // namespace KTensor

#endif // KTENSOR_FUNCTIONS_HPP