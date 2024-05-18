#ifndef KTENSOR_AUXILIARY_HPP
#define KTENSOR_AUXILIARY_HPP

#include <concepts>
#include <array>
#include <tuple>
#include <type_traits>

namespace KTensor{

/********************************************
 *    Aliases for frequently-used types     *
 ********************************************/

// Basic integer type
using int_t = std::size_t;

// char sequence
template<char...C>
using chars = std::integer_sequence<char,C...>;

// int_t sequence
template<int_t...I>
using ints = std::integer_sequence<int_t,I...>;

// Zero-based sequence of consecutive int_t's
template<int_t I>
using intrange = std::make_integer_sequence<int_t,I>;

/****************************
 *    Special constants     *
 ****************************/

// Deferred extents, analogous to std::dynamic extent for dynamic extents.
constexpr int_t deferred_extent = KTENSOR_MDSPAN_NAMESPACE::dynamic_extent - 1;

/*********************************************
 *    Concepts and trait-defining structs    *
 *********************************************/

/** A user may define partial specializations of AdditiveIdentity in order
 * to do implicit summation with non-numeric types.
 * @tparam T Any type
*/
template<typename T>
struct AdditiveIdentity{
  constexpr static T value = 0;
};

/** Concept to constrain template parameters to any integral type
 * @tparam T Any type
*/
template <typename T>
concept Int = (std::is_integral_v<T>);

/** Concept to constrain template parameters to any numerical type
 * @tparam T Any type
*/
template<typename T>
concept Numeric = (std::is_arithmetic_v<T>);

/** Type that can be added
 * @tparam T Any type
*/
template<typename T>
concept Addable = requires(T x){
  {x + x} -> std::same_as<T>;
};

/** Type that can be negated
 * @tparam T Any type
 */
template<typename T>
concept Negatable = requires(T x){
  {-x} -> std::same_as<T>;
};

/** Concept for requiring a set of types to all share (i.e. be convertible to)
 * a specified type
 * @tparam T Desired type
 * @tparam U... List of types
 */
template<typename T, typename...U>
concept Multi = (std::convertible_to<U, T> && ...);

/** Alias for extents types
 * So we don't have to type out the namespace or the int_t every time.
 * @tparam X... List of extents
*/
template<int_t...X>
using Extents = KTENSOR_MDSPAN_NAMESPACE::extents<int_t,X...>;

/** Helper structs to determine if a type is an extents instance
 * @tparam T Any type
*/
template<typename T>
struct is_extent{
  constexpr static bool value = false;
};

/** Partial specialization for types that are Extents's
 * @tparam X... List of extents
*/
template<int_t...X>
struct is_extent<Extents<X...>>{
  constexpr static bool value = true;
  constexpr static bool has_deferred_extent = ((X == deferred_extent) || ...);
};

/** Concept for constraining template parameters to extent types
 * @tparam T Any type
*/
template <typename T>
concept Extent_c = (is_extent<T>::value);

/** Concept for constraining template parameters to Extent_t types where none
 * of the extents are deferred.
 * @tparam T Any type
*/
template <typename T>
concept NondeferredExtent_c = (is_extent<T>::value && !is_extent<T>::has_deferred_extent);

/** Forward declaration of the Index class
 * @tparam I Character associated with the Index
 */ 
template<char I>
requires (I != ' ')
class Index;

// Helper struct to identify Index types for use in defining concepts
template<typename T>
struct is_Index{
  constexpr static bool value = false;
};

// Partial specialization for types that are Indices
template<char I>
struct is_Index<Index<I>>{
  constexpr static bool value = true;
};

// Concept to constrain template parameters to Index types
template <typename T>
concept Index_c = (is_Index<T>::value);

// Concept to constrain template parameters to Index or integer types
template <typename T>
concept IntOrIndex = (is_Index<T>::value || std::is_integral_v<T>);

/** Forward declaration of the IndexMap class
 * @tparam X Type describing the extents for the indices involved
 * in the map
 * T... A list of integer or Index types
*/
template<Extent_c X, IntOrIndex...T>
requires(X::rank() == sizeof...(T))
struct IndexMap;

// Helper struct to identify IndexMap types
template<typename T>
struct is_IndexMap{
  constexpr static bool value = false;
};

// Partial specialization for types that are IndexMaps
template<Extent_c E, IntOrIndex...T>
struct is_IndexMap<IndexMap<E, T...>>{
  constexpr static bool value = true;
};

// Concept to constrain templates to accept only IndexMap types
template <typename T>
concept IndexMap_c = (is_IndexMap<T>::value);

/** Forward declaration of the TensorExpression class
 * @tparam C Type of the wrapped object
 * @tparam M IndexMap type for the object
 * @tparam I... Indices involved in this expression, with repetitions
*/
template<typename C, IndexMap_c M, char...I>
class TensorExpression;

// Helper structs to determine if a type is a TensorExpression
template<typename T>
struct is_expr{
  constexpr static bool value = false;
};

// Partial specialization for types that are TensorExpressions
template<typename C, IndexMap_c M, char...I>
struct is_expr<TensorExpression<C,M,I...>>{
  constexpr static bool value = true;
};

// Concept to constrain template parameters to TensorExpressions
template<typename T>
concept TensorExpr_c = (is_expr<T>::value || is_Index<T>::value);

// Concept to constraint template parameters to types that are not
// TensorExpressions, i.e. that are scalars.
template<typename T>
concept NonExpr_c = (!TensorExpr_c<T>);


/*****************************
 *     Utility functions     *
 *****************************/

/** Return the passed integer or zero if the argument is not an integer.
 * Used to set the fixed arguments when a subscript has mixed integers and
 * Indices.
 * @tparam T Type of the argument
 * @arg T q Anything
 * @return If q is an integer type, return q. Otherwise return integer 0.
 */
template<IntOrIndex T>
constexpr int_t int_from_int_or_Index(T q){
  if constexpr(std::is_integral_v<T>){
    return static_cast<int_t>(q);
  }else{
    return 0;
  }
}

/** If the template parameter is an Index with some character as its template
 * parameter, return that character. Otherwise return character ' '.
 * Used to identify the index characters when a subscript has mixed integers
 * and Indices.
 * @tparam T Any type descriptor
 * @return I if T is an instance of Index<I>, otherwise the character ' '
 */
template<IntOrIndex T>
consteval char char_from_int_or_Index(){
  if constexpr(std::is_integral_v<T>){
    return ' ';
  }else{
    return T::symbol;
  }
}

/** Return the position of a character in a parameter pack
 * @tparam I... Characters to search through
 * @arg i Character to search for
 * @return Zero-based index of i in the pack I
 * @note This function cannot be consteval because it is used in the dynamic
 * extents lambdas which cannot be consteval or constexpr.
 */
template<char...I>
constexpr int_t char_location_in_set(char i){
  int_t idx = 0;
  if (((I == i ? true : (++idx, false)) || ...)){
    return idx;
  }else{
    return -1;
  }
}

/**********************************************************
 *    Classes and helper structs for index operations     *
 **********************************************************/

// Helper struct to obtain an extents type from an integer sequence
template<typename T>
struct ExtentsFromSeqType{};

template<int_t...D>
struct ExtentsFromSeqType<ints<D...>>{
  using type = Extents<D...>;
};

/** Identify the Index characters for each Index<I> type among a set of types
 * and extract the dimensions for each.
 * @tparam X Extents type
 * @tparam T... A sequence of integer or Index<I> types
 */
template<Extent_c X, IntOrIndex...T>
requires(X::rank() == sizeof...(T))
struct IndexMap{
  private:
    constexpr static int_t nargs = sizeof...(T);
    constexpr static int_t nints = (0 + ... + (std::is_integral_v<T> ? 1 : 0));
    constexpr static int_t nindices = nargs - nints;

    consteval static auto setup(){
      constexpr std::array<bool, nargs> is_int = {{std::is_integral_v<T>...}};
      constexpr std::array<char, nargs> indices = {{char_from_int_or_Index<T>()...}};
      std::array<char,nindices> index_chars{};
      std::array<int_t,nindices> index_locations{};
      std::array<int_t,nindices> index_dims{};

      int_t idx = 0;
      for(int_t i=0; i<nargs; ++i){
        if (!is_int[i]){
          index_locations[idx] = i;
          index_dims[idx] = X::static_extent(i);
          index_chars[idx] = indices[i];
          ++idx;
        }
      }
      return std::tuple{index_chars, index_locations, index_dims};
    };
    constexpr static auto data = setup();

  public:
    constexpr static std::array<char,nindices> index_chars = std::get<0>(data);
    constexpr static std::array<int_t,nindices> index_locations = std::get<1>(data);
    constexpr static std::array<int_t,nindices> index_extents = std::get<2>(data);
    constexpr static bool all_integer = (nindices == 0);

  private:
    /** The index characters as a sequence */
    constexpr static auto index_chars_seq = []<int_t...idx>(ints<idx...>){
      return chars<index_chars[idx]...>{};
    }(intrange<nindices>());

    /** The index dimensions as a sequence */
    constexpr static auto dims_seq = []<int_t...idx>(ints<idx...>){
      return std::integer_sequence<int_t, index_extents[idx]...>{};
    }(intrange<nindices>());

    /** The index dimensions as a type */
    using dims_t = typename std::remove_const<decltype(dims_seq)>::type;

  public:
    /** The index characters as a type */
    using index_chars_t = typename std::remove_const<decltype(index_chars_seq)>::type;

    /** Extents corresponding to the indices */
    using extents_t = typename ExtentsFromSeqType<dims_t>::type;

    /** Rank of the tensor expression formed by subscripting
     * @note This rank differs from that of the Tensor from which it is formed
     * if some of the Tensor subscripts are integers.
    */
    constexpr static int_t rank = extents_t::rank();

    /** Get the position of a character among those of Indices associated
     * with this map.
     * @param c Character whose index is to be found
     * @return Position of the character among the indices involved in this map
    */
    consteval static int_t char_location_in_map(char const c){
      for(int_t i=0; i<nindices; ++i){
        if (index_chars[i] == c){
          return i;
        }
      }
      return -1;
    }
};

/** Identify the indices present in a map but not present in a given set, and
 * extract the dimensions for those indices.
 * @tparam M IndexMap type
 * @tparam P... Index characters to exclude
 */
template<IndexMap_c M, char...P>
struct ImplicitSummationMap{
  private:
    constexpr static int_t nI = M::rank;
    constexpr static int_t nP = sizeof...(P);

    consteval static auto setup(){
      std::array<char,nP> Pchars = {{P...}};

      std::array<char,nI> repeated_chars{};
      std::array<int_t,nI> repeated_char_static_extents{};
      std::array<int_t,nI> repeated_char_dynamic_extent_locs{};

      int_t nunique = 0;

      for(int_t i=0; i<nI; ++i){
        if (M::index_extents[i] == deferred_extent){
          // If an extent is deferred, the extent to use
          // will be inferred from a later appearance of the
          // same index.
          continue;
        }
        bool found = false;
        char const ic = M::index_chars[i];
        // Check that this ic has not already been found
        // Prevents duplicates caused by repeated indices
        for(int_t j=0; j<nunique && !found; ++j){
          if (ic == repeated_chars[j]){
            found = true;
          }
        }
        if (!found){
          // Check that this ic is not among the P indices
          for(int_t j=0; j<nP && !found; ++j){
            if (ic == Pchars[j]){
              found = true;
            }
          }
          if (!found){
            repeated_chars[nunique] = ic;
            repeated_char_static_extents[nunique] = M::index_extents[i];
            repeated_char_dynamic_extent_locs[nunique] = i;
            ++nunique;
          }
        }
      }

      return std::tuple{nunique,
                        repeated_chars,
                        repeated_char_static_extents,
                        repeated_char_dynamic_extent_locs};
    }

    constexpr static auto data = setup();
    constexpr static int_t nunique = std::get<0>(data);

  public:
    constexpr static std::array<char,nI> repeated_index_chars = std::get<1>(data);
    constexpr static std::array<int_t,nI> repeated_index_extents = std::get<2>(data);
    constexpr static std::array<int_t,nI> repeated_index_dynamic_extent_locs = std::get<3>(data);

  private:
    /** The repeated index dimensions as a std::integer_sequence<int>() */
    constexpr static auto repeated_index_extents_seq = []<int_t...idx>(ints<idx...>){
      return ints<repeated_index_extents[idx]...>{};
    }(intrange<nunique>());

    /** The repeated and passed indices as a std::integer_sequence<char>() */
    constexpr static auto specified_indices_array = []<int_t...idx>(ints<idx...>){
      return std::array<char,nI+nP> {{repeated_index_chars[idx]..., P...}};
    }(intrange<nunique>());

    /** The repeated index dimensions as a type */
    using repeated_index_dims_t = typename std::remove_const<decltype(repeated_index_extents_seq)>::type;

  public:
    using repeated_index_extents_t = typename ExtentsFromSeqType<repeated_index_dims_t>::type;

    /** Get the position of a character among those of the repeated and passed Indices
     * @param c Character to search for
     * @return Zero-based index of the character
    */
    consteval static int_t char_location_in_passed(char const c){
      for(int_t i=0; i<nI+nP; ++i){
        if (c == specified_indices_array[i]){
          return i;
        }
      }
      return -1;
    }
};

template<typename M, typename B>
struct JoinMaps{};

/** Helper struct to join two IndexMaps into one
 * @tparam DM... Extents from the first map
 * @tparam IM... Index characters from the first map
 * @tparam DB... Extents from the second map
 * @tparam IB... Index characters from the second map
 * 
*/
template<int_t...DM, IntOrIndex...IM, int_t...DB, IntOrIndex...IB>
struct JoinMaps<IndexMap<Extents<DM...>,IM...>,
                IndexMap<Extents<DB...>,IB...>>
{
  // Concatenate the extents and index characters into a new IndexMap
  using type = IndexMap<Extents<DM...,DB...>, IM..., IB...>;
};

template<typename...T>
struct JoinMapsFromTypes{
  // Default to empty extents for a non-tensor, i.e. scalar expression type
  using type = IndexMap<Extents<>>;
};

template<TensorExpr_c T>
struct JoinMapsFromTypes<T>{
  using type = T::map_t;
};

/** Join IndexMaps from several types, which may be scalars or TensorExpressions,
 * into a single IndexMap
 * @tparam T1 First type
 * @tparam T2 Second type
 * @tparam Ts... Additional types
*/
template<typename T1, typename T2, typename...Ts>
struct JoinMapsFromTypes<T1,T2,Ts...>{
  using type = typename JoinMaps<typename JoinMapsFromTypes<T1>::type,
                                 typename JoinMapsFromTypes<T2,Ts...>::type>::type;
};

} // namespace KTensor
#endif //KTENSOR_AUXILIARY_HPP