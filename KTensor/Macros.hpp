#ifndef KTENSOR_MACROS_HPP
#define KTENSOR_MACROS_HPP

/** Return a TensorExpression from a subscripting of a Tensor
 * @tparam T... Types of the arguments
 * @arg Is... Integers or Index arguments
 * @return TensorExpression representing a segment of this Tensor
 * The weird lambda construction allows us to keep the last template parameter
 * of TensorExpression as a pack of chars, instead of a type chars<I...>,
 * which means we can define TensorExpression as the primary template
 * instead of a partial specialization.
*/
#define FormTensorExpression(Qualifier) \
template<IntOrIndex...T> \
auto operator()(const T... Is) Qualifier \
requires((sizeof...(Is) == rank)) \
{ \
  std::array<int_t,rank> const fixed_args = {{int_from_int_or_Index(Is)...}}; \
  using map_t = IndexMap<extents_t, T...>; \
  return [this, &fixed_args]<char...idx>(chars<idx...>){ \
    return TensorExpression<tensor_type, map_t, idx...>(*this, fixed_args); \
  }(typename map_t::index_chars_t()); \
}

#define BasicDynamicExtentLambda \
auto dynamic_extent_lambda = [this](int_t const n) -> int_t { \
  return this->extent(n); \
};

#define PermutingDynamicExtentLambda \
auto dynamic_extent_lambda = [this](int_t const n) -> int_t { \
  return this->extent(ISM_t::repeated_index_dynamic_extent_locs[n]); \
};

/** Constructor
 * @arg contents_ Underlying tensor wrapped by this object
 * @arg fixed_args_ std::array containing the arguments where they are
 *      integers and zeros elsewhere. For example, if this object was
 *      created from an expression like A(1,j,k,3) then fixed_args_ would be
 *      {{1,0,0,3}}.
*/
#define TensorExpressionConstructor(ContentsType) \
constexpr TensorExpression(ContentsType contents_, std::array<int_t,rank> const fixed_args_) \
                         : contents(contents_), fixed_args(fixed_args_){}

/** Forward a request for the dynamic extent to the underlying Tensor
 * @arg n Index for which to get the extent.
 * @return Dynamic extent for the index.
*/
#define BasicExtentFunction \
int_t extent(int_t const n) const{ \
  return contents.extent(M::index_locations[n]); \
}

/** Check if a bottom-level tensor expression involves a contraction over a given index
 * @arg i Index character to test
 * @return True iff the expression involves a contraction over the index j
*/
#define BasicContractionCheck \
consteval static bool contracting(char i){ \
  int const count = (0 + ... + ((i == I) ? 1 : 0)); \
  return (count > 1); \
}

/** Return a reference to the element corresponding to a given sequence
 * of subscripts
 * @arg v... Integer subscript values
 * @return Writeable reference to the tensor element at the specified position
 * The Ret macro argument allows to return a reference or a value as desired.
 */
#define AllIntegerSubscriptOperator(ReturnType) \
ReturnType subscript(const Multi<int_t> auto...v) const \
requires(sizeof...(v) == sizeof...(I)) \
{ \
  int_t i=0; \
  std::array<int_t, rank> all_args = fixed_args; \
  (..., (all_args[M::index_locations[i++]] = v)); \
  return contents.subscript(all_args); \
}

/** Sum over repeated indices of this expression
 * @tparam P... Characters for the indices whose values are specified
 * @arg Ps... Values corresponding to the indices P
 * @return Result of the implicit summation
 */
#define ImplicitSummationFunction \
template<char...P> \
return_type implicit_summation(chars<P...>, const Multi<int_t> auto...Ps) const \
requires(Addable<return_type>) \
{ \
  using ISM_t = ImplicitSummationMap<map_t, P...>; \
  return_type result = AdditiveIdentity<return_type>::value; \
  auto repeated_index_lambda = [this, &result, Ps...](const Multi<int_t> auto... r){ \
    const std::array<int_t,sizeof...(r)+sizeof...(P)> passed_args = {{r..., Ps...}}; \
    result += this->subscript(passed_args[ISM_t::char_location_in_passed(I)]...); \
  }; \
  PermutingDynamicExtentLambda \
  Loop<typename ISM_t::repeated_index_extents_t>()(dynamic_extent_lambda, repeated_index_lambda); \
  return result; \
}

#define ReduceToScalar \
explicit(!map_t::all_integer) operator return_type() const{ \
  static_assert(ReducibleToScalar<expr_t, I...>::value, "EXPRESSION DOES NOT REDUCE TO A SCALAR"); \
  return implicit_summation(chars<>()); \
}

/** Define a partial specialization of TensorExpression for a constant tensor
 * @tparam S Underlying scalar type
 * @tparam D Dimension of the Kronecker delta
 * @tparam M IndexMap type
 * @tparam I... Index characters
 */
#define ConstantTensorSpecialization(TensorType) \
template<typename S, int_t D, IndexMap_c M, char...I> \
class TensorExpression<TensorType<S,D>,M,I...> \
{ \
  public: \
    using return_type = S; \
    using map_t = M; \
  private: \
    using contents_t = TensorType<S,D>; \
    using expr_t = TensorExpression<contents_t,M,I...>; \
    constexpr static int_t rank = contents_t::rank; \
    contents_t const& contents; \
    std::array<int_t,rank> const fixed_args; \
  public: \
    TensorExpressionConstructor(contents_t const&) \
    BasicExtentFunction \
    BasicContractionCheck \
    AllIntegerSubscriptOperator(S) \
    ImplicitSummationFunction \
    ReduceToScalar \
};

#endif // KTENSOR_MACROS_HPP