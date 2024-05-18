#ifndef KTENSOR_NESTED_LOOPS_HPP
#define KTENSOR_NESTED_LOOPS_HPP

namespace KTensor{

/** Struct for inner (i.e. not the outermost) loops in a sequence of nested for loops
 * @tparam N Index of the extent for which a loop is to be formed
 * @tparam X Type describing the extents of each loop with the innermost
 *           loop being last.
 */
template <int_t N, Extent_c X>
struct ILoop{
  private:
    static constexpr int_t rank = X::rank();

  public:

    /** Evaluate a function f at all combinations of values
     * @tparam ExprExtentFunction Type of a lambda returning dynamic extent values
     * @tparam F Type of the function to evaluate inside the innermost loop
     * @tparam Args... Types of the input arguments
     * @arg ext Lambda returning dynamic extent values
     * @arg f The function to evaluate inside the innermost loop
     */
    template <typename ExprExtentFunction, typename F, typename... Args>
    constexpr void operator()(ExprExtentFunction& ext, F& f, Args... args) {
      if constexpr(X::static_extent(N) != KTENSOR_MDSPAN_NAMESPACE::dynamic_extent){
        // Use the static extent known at compile time
        for (int_t i=0; i < X::static_extent(N); ++i) {
          if constexpr (N < rank-1){
            ILoop<N+1,X>()(ext, f, args..., i);
          }else{
            // This loop is the innermost so just evaluate the function
            f(args..., i);
          }
        }
      }else{
        // Pull the dynamic extent from the underlying Tensor
        int_t const ext_N = ext(N);
        for (int_t i=0; i < ext_N; ++i) {
          if constexpr (N < rank-1){
            ILoop<N+1,X>()(ext, f, args..., i);
          }else{
            // Inside the innermost loop
            f(args..., i);
          }
        }
      }
   }
};

/** Struct for the outermost of the nested loops
 * @tparam X Type describing the extents of each loop with the innermost
 *           loop being last.
*/
template<Extent_c X>
struct Loop{
  /** Evaluate a function f at all combinations of values
   * @tparam ExprExtentFunction Type of a lambda returning dynamic extent values
   * @tparam F Type of the function to evaluate inside the innermost loop
   * @arg ext Lambda returning dynamic extent values
   * @arg f The function to evaluate
   */
  template <typename ExprExtentFunction, typename F>
  constexpr void operator()(ExprExtentFunction& ext, F& f){
    if constexpr (X::rank() > 0){
      ILoop<0,X>()(ext, f);  // Calls operator() of ILoop using the function f
    }else{
      // Default case for no looping
      f();
    }
  }
};
} // namespace KTensor

#endif //KTENSOR_NESTED_LOOPS_HPP