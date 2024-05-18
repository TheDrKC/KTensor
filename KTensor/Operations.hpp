#ifndef KTENSOR_OPERATIONS_HPP
#define KTENSOR_OPERATIONS_HPP

namespace KTensor{

/** Define return types for operations
 * This is really only needed to force integer division to return doubles
 * @tparam S1 First scalar type
 * @tparam S2 Second scalar type
 * @tparam OpChar Character indicating the operation being performed
 * The member type is std::common_type<S1,S2> for all combinations except those
 * involving two integer types and division, for which the return type is double
*/
template<typename S1, typename S2, char OpChar>
struct BinaryOpReturn{
  constexpr static bool defined = false;
};

#define ReturnSpecialization(Op,OpChar) \
template<typename S1, typename S2> \
struct BinaryOpReturn<S1,S2,OpChar>{ \
  using type = decltype(std::declval<S1>() Op std::declval<S2>()); \
  constexpr static bool defined = true; \
};

ReturnSpecialization(+,'+')
ReturnSpecialization(-,'-')
ReturnSpecialization(*,'*')
ReturnSpecialization(/,'/')

// Enforce floating-point division for integers
template<typename S1, typename S2>
requires(std::is_integral_v<S1> && std::is_integral_v<S2>)
struct BinaryOpReturn<S1,S2,'/'>{
  using type = double;
  constexpr static bool defined = true;
};

template<char Op, typename A, typename B>
class BinaryOperator{};

/** Representation of an operation on two TensorExpressions
 * @tparam Op Character '*', '/', '+', or '-' indicating the operation
 * @tparam A Type wrapped by the first expression
 * @tparam MA IndexMap type for the first expression
 * @tparam I... Index characters involved in the first expression
 * @tparam B Type wrapped by the second expression
 * @tparam MB IndexMap type for the second expression
 * @tparam J... Index characters involved in the second expression
 */
template<char Op, typename A, IndexMap_c MA, char...I,
                  typename B, IndexMap_c MB, char...J>
class BinaryOperator<Op, TensorExpression<A,MA,I...>,
                         TensorExpression<B,MB,J...>>{

  private:
    using LHS_expr_t = TensorExpression<A,MA,I...>;
    using RHS_expr_t = TensorExpression<B,MB,J...>;
    LHS_expr_t lhs;
    RHS_expr_t rhs;

  public:
    using return_type = BinaryOpReturn<typename LHS_expr_t::return_type,
                                       typename RHS_expr_t::return_type,Op>::type;
    // Form a new IndexMap type for the expression as a whole, using those
    // of the two operands.
    using map_t = typename JoinMaps<MA,MB>::type;

    /** Constructor */
    constexpr BinaryOperator(LHS_expr_t const& s1, RHS_expr_t const& s2) : lhs(s1), rhs(s2) {}

    /** Forward a request for the dynamic extent to the wrapped object
     * @arg n Index for which to get the extent.
    */
    int_t extent(int_t const& n) const{
      if (n < sizeof...(I)){
        return lhs.extent(n);
      }else{
        return rhs.extent(n-sizeof...(I));
      }
    }

    consteval static bool contracting(char const i){
      if constexpr((Op == '*') || (Op == '/')){
        if ((false || ... || (i == I)) && (false || ... || (i == J))){
          return true;
        }
      }
      return LHS_expr_t::contracting(i) || RHS_expr_t::contracting(i);
    }

    /** Evaluate the product at a given set of index values
     * @arg Is... Values of the indices corresponding to the chars in I and J
     * @return Result of the operation at the passed index values
     */
    return_type subscript(const Multi<int_t> auto...Is) const
    requires(sizeof...(I)+sizeof...(J) == sizeof...(Is))
    {
      const std::array<int_t,sizeof...(Is)> passed_args = {{Is...}};
      if constexpr (Op == '*'){
        return lhs.subscript(passed_args[map_t::char_location_in_map(I)]...) *
               rhs.subscript(passed_args[map_t::char_location_in_map(J)]...);
      }else if constexpr (Op == '/'){
        return ((return_type) lhs.subscript(passed_args[map_t::char_location_in_map(I)]...)) /
               ((return_type) rhs.subscript(passed_args[map_t::char_location_in_map(J)]...));
      }else if constexpr (Op == '+'){
        return lhs.subscript(passed_args[map_t::char_location_in_map(I)]...) +
               rhs.subscript(passed_args[map_t::char_location_in_map(J)]...);
      }else if constexpr (Op == '-'){
        return lhs.subscript(passed_args[map_t::char_location_in_map(I)]...) -
               rhs.subscript(passed_args[map_t::char_location_in_map(J)]...);
      }
    }

    /** Sum over repeated indices of this expression
     * @tparam P... Free index characters
     * @arg Ps... Values corresponding to the indices P
     * @return Result of the implicit summation
     */
    template<char...P>
    return_type implicit_summation(chars<P...>, const Multi<int_t> auto...Ps) const
    requires(Addable<return_type>)
    {
      if constexpr(Op == '*' || Op == '/'){
        // Get the set complement of P in IuJ and the corresponding dimensions
        using ISM_t = ImplicitSummationMap<map_t, P...>;

        return_type result = AdditiveIdentity<return_type>::value;

        // Is = values of the repeated indices being summed over
        auto repeated_index_lambda = [this, &result, Ps...](const Multi<int_t> auto... Is){
          const std::array<int_t,sizeof...(Is)+sizeof...(P)> passed_args = {{Is..., Ps...}};
          if constexpr(Op == '*'){
            result += lhs.subscript(passed_args[ISM_t::char_location_in_passed(I)]...) *
                      rhs.subscript(passed_args[ISM_t::char_location_in_passed(J)]...);
          }else{
            result += ((return_type) lhs.subscript(passed_args[ISM_t::char_location_in_passed(I)]...)) /
                      ((return_type) rhs.subscript(passed_args[ISM_t::char_location_in_passed(J)]...));
          }
        };

        PermutingDynamicExtentLambda

        Loop<typename ISM_t::repeated_index_extents_t>()(dynamic_extent_lambda, repeated_index_lambda);
        return result;
      }else{
        // Addition or subtraction
        if constexpr (Op == '+'){
          return lhs.implicit_summation(chars<P...>(), Ps...) +
                 rhs.implicit_summation(chars<P...>(), Ps...);
        }else{
          return lhs.implicit_summation(chars<P...>(), Ps...) -
                 rhs.implicit_summation(chars<P...>(), Ps...);
        }
      }
    }
};

/** Operators to form a BinaryOperator from two TensorExpressions, or from a
 * TensorExpression and a scalar
 * @tparam A Type wrapped by the first expression
 * @tparam MA IndexMap type for the first expression
 * @tparam I... Index characters involved in the first expression
 * @tparam B Type wrapped by the second expression
 * @tparam MB IndexMap type for the second expression
 * @tparam J... Index characters involved in the second expression
 * @arg s1 First factor, a TensorExpression<A,MA,I...>
 * @arg s2 Second factor, a TensorExpression<B,MB,J...>
 * @return A TensorExpression wrapping the object representing the result
 * @note Since a/b = a*(1/b), the quotient is really a product. Therefore
 *       implicit summation will be applied across '/'.
 */
#define BinaryOp(Op, OpChar, Requirement) \
template<typename A, IndexMap_c MA, char...I, typename B, IndexMap_c MB, char...J> \
requires(Requirement<MA,MB>::value && \
        BinaryOpReturn<typename A::return_type, \
                       typename B::return_type, OpChar>::defined) \
auto operator Op (TensorExpression<A,MA,I...> const& s1, \
                  TensorExpression<B,MB,J...> const& s2){ \
  using Result_t = const BinaryOperator<OpChar, TensorExpression<A,MA,I...>, \
                                                TensorExpression<B,MB,J...>>; \
\
  return TensorExpression<Result_t, typename Result_t::map_t, I..., J...>(Result_t(s1,s2)); \
} \
\
template<NonExpr_c SC, typename A, IndexMap_c M, char...I> \
auto operator Op(TensorExpression<A,M,I...> const& T, SC const& c) \
requires(BinaryOpReturn<typename A::return_type,SC,OpChar>::defined) \
{ \
  using ScalarExpr_t = ScalarExpr<SC>; \
  using Result_t = BinaryOperator<OpChar, TensorExpression<A,M,I...>, ScalarExpr_t>; \
  return TensorExpression<Result_t, M, I...>(Result_t(T,ScalarExpr_t(c))); \
} \
template<NonExpr_c SC, typename A, IndexMap_c M, char...I> \
auto operator Op(SC const& c, TensorExpression<A,M,I...> const& T) \
requires(BinaryOpReturn<SC,typename A::return_type,OpChar>::defined) \
{ \
  using ScalarExpr_t = ScalarExpr<SC>; \
  using Result_t = BinaryOperator<OpChar, ScalarExpr_t, TensorExpression<A,M,I...>>; \
  return TensorExpression<Result_t, M, I...>(Result_t(ScalarExpr_t(c),T)); \
}

BinaryOp(+, '+', FreeIndexDimensionsMatch)
BinaryOp(-, '-', FreeIndexDimensionsMatch)
BinaryOp(*, '*', CommonIndexDimensionsMatch)
BinaryOp(/, '/', CommonIndexDimensionsMatch)

/** Representation of a unary negation of a TensorExpression
 * @tparam A Type wrapped by the TensorExpression
 * @tparam M IndexMap type
 * @tparam I... Index characters involved in the expression
 */
template<typename A, IndexMap_c M, char...I>
requires(Negatable<typename A::return_type>)
class Negation
{
  private:
    using expr_t = TensorExpression<A, M, I...>;
    expr_t const& T;
  public:
    using return_type = A::return_type;
    using extents_t = typename M::extents_t;

    /** Constructor
     * @arg T_ TensorExpression being negated
    */
    constexpr Negation(expr_t const& T_) : T(T_){}

    /** Forward a request for the dynamic extent to the negated expression
     * @arg n Index for which to get the extent.
    */
    int_t extent(int_t const& n) const{
      return T.extent(n);
    }

    consteval static bool contracting(char const i){
      return expr_t::contracting(i);
    }

    /** Return the negative of the wrapped expression at a set of index values
     * @arg Is... Values of the indices in template parameter I...
     * @return Negative of the contents at the passed index values
     */
    return_type subscript(const Multi<int_t> auto...Is) const{
      return -T.subscript(Is...);
    }

    /** Sum over repeated indices of this expression
     * @tparam P... Free index characters
     * @arg Ps... Values corresponding to the indices P
     * @return Result of the implicit summation
     */
    template<char...P>
    requires(Addable<return_type>)
    return_type implicit_summation(chars<P...>, const Multi<int_t> auto...Ps) const
    {
      return -T.implicit_summation(chars<P...>(), Ps...);
    }
};

/** Operator for unary negation
 * @tparam A Type wrapped by the TensorExpression
 * @tparam M IndexMap type
 * @tparam I... Index characters involved in the expression
 * @arg T Tensor expression
 * @return A TensorExpression wrapping the object representing the negation
 */
template<typename A, IndexMap_c M, char...I>
auto operator-(TensorExpression<A,M,I...> const& T)
requires(Negatable<typename A::return_type>)
{
  using Result_t = Negation<A,M,I...>;
  return TensorExpression<Result_t, M, I...>(Result_t(T));
}

} // namespace KTensor

#endif  // KTENSOR_OPERATIONS_HPP