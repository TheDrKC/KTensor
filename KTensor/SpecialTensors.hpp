#ifndef KTENSOR_SPECIAL_TENSORS_HPP
#define KTENSOR_SPECIAL_TENSORS_HPP

namespace KTensor{

/** Base class for Kronecker delta tensors
 * @tparam S Scalar type returned by subscripting
 * @tparam D Number of dimensions, which currently must be 2
*/
template<Numeric S, int_t D=2>
requires(D==2)  // May generalize to higher dimensions in the future
class KroneckerDelta{
  private:
    using tensor_type = KroneckerDelta<S,D>;
  public:
    using extents_t = Extents<deferred_extent,deferred_extent>;
    using return_type = S;
    constexpr static int_t rank = D;

    constexpr KroneckerDelta(){}

    /** Getter function for this object's extent
     * @arg n Index for which to get the extent
     * @return The extent (same for all indices n)
     * @note This function should never be called because there is no reason to
     * check the extents of a KroneckerDelta, and no KroneckerDelta will ever
     * be on the LHS of an assignment therefore will never control extents of
     * a set of loops. This function needs to be defined, however, to prevent
     * compiler errors.
     */
    constexpr static int_t extent(int_t const n) {
      return deferred_extent;
    }

    /** Evaluate the delta at two integer arguments
     * @tparam IntType Integer type of both arguments
     * @arg i First argument
     * @arg j Second argument
     * @return 1 if i==j, 0 otherwise
    */
    template<Int IntType>
    constexpr S subscript(IntType const& i, IntType const& j) const {
      return (i==j) ? 1 : 0;
    }

    /** Evaluate the delta at two integer arguments given as a single std::array
     * @tparam IntType Integer type of the array
     * @arg ij Array containing the two arguments
     * @return 1 if ij[0]==ij[1], 0 otherwise
    */
    template<Int IntType>
    constexpr S subscript(std::array<IntType,2> const& ij) const {
      return this->subscript(ij[0], ij[1]);
    }

    FormTensorExpression(const)
};

ConstantTensorSpecialization(KroneckerDelta)

/** 3x3x3 Levi-Civita symbol
 * Example usage:
 *    LeviCivita<int,3> epsilon;
 *    auto uxv(i) = epsilon(i,j,k)*u(j)*v(k);
 * @tparam S underlying scalar type
 * @tparam D extent of each dimension (currently must be 3)
 */
template<Numeric S, int_t D=3>
requires((D==3))  // May generalize to higher dimensions in the future
class LeviCivita{
  private:
    constexpr static int_t extent_m = D;
    using tensor_type = LeviCivita<S,D>;
  public:
    constexpr static int_t rank = D;
    using return_type = S;
    using extents_t = Extents<D,D,D>;

    /** Getter function for this object's extent
     * @arg n Index for which to get the extent
     * @return The extent (same for all indices n)
     */
    constexpr static int_t extent(int_t const n) {
      return extent_m;
    }

    /** Subscript operator
     * @arg Is... Subscript values
     * @return The i,j,k element of the rank-3 Levi-Civita symbol where Is... = i,j,k
     */
    constexpr S subscript(const Multi<int_t> auto...Is) const
    requires(sizeof...(Is) == rank)
    {
      const std::array<int_t,rank> passed_args = {{Is...}};
      const int_t i = passed_args[0];
      const int_t j = passed_args[1];
      const int_t k = passed_args[2];
      return (i==j || j==k || i==k) ? 0 : ((i+1)%3 == j ? 1 : -1);
    }

    /** Subscript operator taking an array
     * @arg args std::array of subscript values
     * @return The i,j,k element of the rank-3 Levi-Civita symbol where args = {{i,j,k}}
     */
    constexpr S subscript(std::array<int_t,D> const& args) const{
      return this->subscript(args[0], args[1], args[2]);
    }

    FormTensorExpression(const)
};

ConstantTensorSpecialization(LeviCivita)

} // namespace KTensor

#endif // KTENSOR_SPECIAL_TENSORS_HPP