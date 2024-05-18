#ifndef KTENSOR_CHECKS_HPP
#define KTENSOR_CHECKS_HPP

namespace KTensor{

/** Check that indices which occur multiple times in a set all have the same
 * dimensions. This check is used to test if an internal contraction is valid.
 * @tparam M IndexMap type
 */
template<IndexMap_c M>
struct RepeatedIndexDimensionsMatch
{
  private:
    /** Setup function to initialize the member flag
     * @return Flag indicating matched dimensions among repeated indices
     */
    consteval static bool checkDimensions(){
      constexpr int_t size = M::rank;

      for(int_t i=0; i<size; ++i){
        int_t const ix = M::index_extents[i];
        char const ic = M::index_chars[i];
        bool const ifinite = (ix != deferred_extent);

        for(int_t j=i+1; j<size; ++j){
          int_t const jx = M::index_extents[j];
          char const jc = M::index_chars[j];
          bool const jfinite = (jx != deferred_extent);

          if (ic == jc){
            if (ifinite == jfinite){
              // Either both extents are finite, or both are deferred
              if (ifinite){
                // Both are finite
                if (ix != jx){
                  return false;
                }
              }else{
                // Both are deferred. This is not OK.
                return false;
              }
            }
          }
        }
      }
      return true;
    }
  public:
    /** Flag indicating matched dimensions */
    constexpr static bool value = checkDimensions();
    static_assert(value, "MISMATCHED INDEX EXTENTS FOR IMPLICIT SUMMATION WITHIN A TENSOR"); 
};

/** Check that any indices which appear in both of two sets have the same
 * dimensions in each. This check is used to test if an external contraction
 * is valid.
 * @tparam MA IndexMap type for the first set
 * @tparam MB IndexMap type for the second set
 */
template<IndexMap_c MA, IndexMap_c MB>
struct CommonIndexDimensionsMatch{
  private:
    /** Setup function to initialize the member flag
     * @return Flag indicating matched dimensions among common indices
     */
    consteval static bool checkDimensions(){
      constexpr int_t lhs_rank = MA::rank;
      constexpr int_t rhs_rank = MB::rank;

      for(int_t i=0; i<lhs_rank; ++i){
        int_t const ix = MA::index_extents[i];
        bool const ifinite = (ix != deferred_extent);
        char const ic = MA::index_chars[i];
        for(int_t j=0; j<rhs_rank; ++j){
          if (ic == MB::index_chars[j]){
            int_t const jx = MB::index_extents[j];
            bool const jfinite = (jx != deferred_extent);

            if (ifinite && jfinite){
              if (ix != jx){
                return false;
              }
            }
          }
        }
      }
      return true;
    }

  public:
    /** Flag indicating matched dimensions */
    constexpr static bool value = checkDimensions();
    static_assert(value,"AN INDEX HAS MISMATCHED EXTENTS");
};

/** Check that any indices that appear exactly once in each of two index sets
 * have the same dimensions in both sets. This check is used to test if a sum
 * of two TensorExpressions is valid.
 * @tparam MA IndexMap type for the first set
 * @tparam MB IndexMap type for the second set
 */
template<IndexMap_c MA, IndexMap_c MB>
struct FreeIndexDimensionsMatch{
  private:
    /** Setup function to initialize the member flag
     * @return Flag indicating matched dimensions among free indices
     */
    consteval static bool checkDimensions(){
      constexpr int_t lhs_rank = MA::rank;
      constexpr int_t rhs_rank = MB::rank;

      for(int_t i=0; i<lhs_rank; ++i){
        char const ic = MA::index_chars[i];

        int_t icount = 0;
        for(int_t i2=0; i2<lhs_rank; ++i2){
          if (ic == MA::index_chars[i2]){
            icount += 1;
          }
        }
        if (icount == 1){
          int_t jcount = 0;
          bool extents_match = true;
          for(int_t j=0; j<rhs_rank; ++j){
            if (ic == MB::index_chars[j]){
              jcount += 1;
              int_t const ix = MA::index_extents[i];
              int_t const jx = MB::index_extents[j];
              if (ix == deferred_extent){
                extents_match = true;
              }else if (jx == deferred_extent){
                extents_match = true;
              }else if (ix != jx){
                extents_match = false;
              }
            }
          }

          if ((jcount == 1) && !extents_match){
            return false;
          }
        }
      }
      return true;
    }
  public:
    /** Flag indicating matched dimensions */
    constexpr static bool value = checkDimensions();
    static_assert(value, "FREE INDEX EXTENTS DO NOT MATCH");
};

/** Check that indices on the RHS which are not on the LHS are involved in
 * implicit summations.
 * @tparam LHS_map_t IndexMap of the LHS expression
 * @tparam RHS_expr_t RHS expression type
 * @tparam J... Index characters on the RHS
*/
template<IndexMap_c LHS_map_t, TensorExpr_c RHS_expr_t, char...J>
struct NonLHSIndicesAreContracting{
  private:
    consteval static auto check_index(char const j){
      if(LHS_map_t::char_location_in_map(j) != (int_t)(-1)){
        return true;
      }else{
        return RHS_expr_t::contracting(j);
      }
    }

  public:
    constexpr static bool value = (true && ... && check_index(J));
    static_assert(value, "AN INDEX ON THE RHS IS NOT ON THE LHS AND IS NOT IMPLICITLY SUMMED OVER");
};

/** Check that indices that are implicitly summed over do not all have
 * deferred extents
 * @tparam LHS_map_t IndexMap of the LHS expression
 * @tparam RHS_expr_t RHS expression type
 * @tparam J... Index characters on the RHS
*/
template<IndexMap_c LHS_map_t, TensorExpr_c RHS_expr_t, char...J>
struct ContractingIndicesAreFinite{
  private:
    consteval static auto check_index(char const j){
      if(LHS_map_t::char_location_in_map(j) != (int_t)(-1)){
        return true;
      }else{
        if (RHS_expr_t::contracting(j)){
          int_t idx = 0;
          return (false || ... || ((j == J) ? (RHS_expr_t::map_t::extents_t::static_extent(idx++) != deferred_extent)
                                            : (++idx,false)));
        }else{
          return true;
        }
      }
    }

  public:
    constexpr static bool value = (true && ... && check_index(J));
    static_assert(value, "COULD NOT INFER EXTENT FOR IMPLICIT SUMMATION");
};

/** Check if an expression can be reduced to a scalar
 * @tparam T TensorExpression type
 * @tparam I... Indices involved in the expression
*/
template<TensorExpr_c T, char...I>
struct ReducibleToScalar{
  constexpr static bool value = (true && ... && T::contracting(I));
};
} // namespace KTensor

#endif // KTENSOR_CHECKS_HPP