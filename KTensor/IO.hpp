#ifndef KTENSOR_OUTPUT_HPP
#define KTENSOR_OUTPUT_HPP

#include <iostream>

namespace KTensor{

/** Allow insertion of an MDTensor into a stream, in particular std::cout
 * @tparam T... Template parameters for an MDTensor
 * @arg ostream Output stream
 * @arg A MDTensor to insert into the stream.
 * @return ostream
*/
template<typename...T>
std::ostream& operator<< (std::ostream& stream, MDTensor<T...> const& A){
  using tensor_type = MDTensor<T...>;
  constexpr int_t rankm1 = tensor_type::rank - 1;
  int_t const last_extent_m1 = (tensor_type::extents_t::static_extent(rankm1) 
                              == KTENSOR_MDSPAN_NAMESPACE::dynamic_extent) ?
                              (A.extent(rankm1)-1) : (tensor_type::static_extent(rankm1)-1);

  auto print_lambda = [&A,&stream,last_extent_m1,rankm1](const Multi<int_t> auto... Is){
  stream << A(Is...) << " ";
  // It would be nice to use a fold expression to get the last element
  // of Is, instead of unpacking into an array, but that would unfortunately
  // result in pointless compiler warnings about unused variables.
  std::array<int_t,tensor_type::rank> const passed_args = {{Is...}};
  if (passed_args[rankm1] == last_extent_m1){
    stream << std::endl;
  }
  };

  auto dynamic_extent_lambda = [&A](int_t const n) -> int_t {
    return A.extent(n);
  };

  Loop<typename tensor_type::extents_t>()(dynamic_extent_lambda, print_lambda);

  return stream;
}
} // namespace KTensor

#endif // KTENSOR_OUTPUT_HPP