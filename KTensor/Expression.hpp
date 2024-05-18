#ifndef KTENSOR_EXPRESSION_HPP
#define KTENSOR_EXPRESSION_HPP

#include <array>

#if KTENSOR_RANDOM_INIT
#include <random>
#endif

namespace KTensor{

/**
 * Generic class representing a tensor expression, i.e. an object that returns
 * a scalar when indices are replaced by integers. Objects of this class wrap
 * objects of other classes that represent tensors or operations on them.
 * Accordingly, member functions of this class just forward their arguments to
 * the corresponding member functions of the wrapped class.
 * @tparam C Type of the wrapped object
 * @tparam M IndexMap type for the object
 * @tparam I... Indices involved in this expression, with repetitions
 * @note The characters in I can also be obtained through M; having them as a
 * separate template parameter allows pack-expanding shortcuts that are far
 * easier to read than what would be required if using M alone.
 */
template<typename C, IndexMap_c M, char...I>
class TensorExpression{

  private:
    using contents_t = C;
    using expr_t = TensorExpression<C,M,I...>;
    contents_t contents;

  public:
    using return_type = C::return_type;
    using map_t = M;

    constexpr TensorExpression(C&& contents_) : contents(contents_) {}

    /** Forward a request for the dynamic extent to the wrapped object
     * @arg n Index for which to get the extent.
    */
    int_t extent(int_t const& n) const{
      return contents.extent(M::index_locations[n]);
    }

    /** Forward a request for whether this expression involves imiplicit
     * summation over a given index.
     * @arg i Index character to test
    */
    consteval static bool contracting(char const i){
      return C::contracting(i);
    }

    /** Forward an implicit summation request to the wrapped object
     * @tparam P... Free index characters
     * @arg Ps... Values corresponding to the indices P
     * @return Result of the implicit summation
     */
    template<char...P>
    return_type implicit_summation(chars<P...>, const Multi<int_t> auto...Ps) const
    requires(Addable<return_type>)
    {
      return contents.implicit_summation(chars<P...>(), Ps...);
    }

    /** Forward a subscripting request to the wrapped object
     * @arg Is... Values of the indices at which to evaluate the subscript
     * @return Requested component of the wrapped object
     */
    return_type subscript(const Multi<int_t> auto... Is) const{
      return contents.subscript(Is...);
    }

    ReduceToScalar
};

// An Index is a TensorExpression that wraps this class
class IndexBase{
  public:
    using return_type = int;
};

template<char I>
class TensorExpression<IndexBase, IndexMap<Extents<deferred_extent>,Index<I>>, I>
{
  private:
  public:
    using return_type = int;
    using map_t = IndexMap<Extents<deferred_extent>,Index<I>>;
    constexpr static char symbol = I;

    constexpr TensorExpression(){}

    int_t extent(int_t const& n) const{
      return deferred_extent;
    }

    /** A lone index is never involved in a contraction
     * @arg i Index character - not actually used
    */
    consteval static bool contracting(char const i){
      return false;
    }

    template<char...P>
    return_type implicit_summation(chars<P...>, const Multi<int_t> auto...Ps) const
    requires(Addable<return_type>)
    {
      std::array<int_t,sizeof...(Ps)> const passed_args = {{Ps...}};
      return passed_args[char_location_in_set<P...>(I)];
    }

    return_type subscript(const int_t Is) const{
      return Is;
    }
};

template<char I>
requires (I != ' ')
class Index : public TensorExpression<IndexBase, IndexMap<Extents<deferred_extent>,Index<I>>, I>
{};

template<NonExpr_c S>
class ScalarBase{
  public:
    using return_type = S;
};

template<NonExpr_c S>
class TensorExpression<ScalarBase<S>, IndexMap<Extents<>>>
{
  private:
    S contents;
  public:
    using return_type = S;
    using map_t = IndexMap<Extents<>>;

    constexpr TensorExpression(S const& contents_): contents(contents_){}

    constexpr int_t extent(int_t const& n) const{
      return 0;
    }

    consteval static bool contracting(char const i){
      return false;
    }

    template<char...P>
    constexpr return_type implicit_summation(chars<P...>, const Multi<int_t> auto...Ps) const
    requires(Addable<return_type>)
    {
      return contents;
    }

    constexpr return_type subscript() const{
      return contents;
    }
};

template<NonExpr_c S>
using ScalarExpr = TensorExpression<ScalarBase<S>, IndexMap<Extents<>>>;

/**
 * MDTensor object, an extension of mdspan taking the same template parameters.
 * @tparam S Scalar type of the tensor elements
 * @tparam X Type representing the extents of the tensor
 * @tparam LayoutPolicy Layout policy type for mdspan. Defaults to layout_right
 * @tparam AccessorPolicy Accessor policy type for mdspan. Defaults to the
 *         default accessor.
 */
template<NonExpr_c S, NondeferredExtent_c X,
        typename LayoutPolicy=KTENSOR_MDSPAN_NAMESPACE::layout_right,
        typename AccessorPolicy=KTENSOR_MDSPAN_NAMESPACE::default_accessor<S>>
requires(X::rank() > 0)
class MDTensor : public KTENSOR_MDSPAN_NAMESPACE::mdspan<S, X, LayoutPolicy, AccessorPolicy>{
  private:
    using mdspan_type = KTENSOR_MDSPAN_NAMESPACE::mdspan<S, X, LayoutPolicy, AccessorPolicy>;
    using tensor_type = MDTensor<S, X, LayoutPolicy, AccessorPolicy>;

  public:
    using return_type = S;
    using extents_t = X;
    // Inherit all constructors from the base mdspan type
    using mdspan_type::mdspan_type;
    constexpr static int_t rank = extents_t::rank();

    /** When all arguments to operator() are integers, return a
     * writeable reference to the underlying element.
     * @tparam T... Types of the arguments, either int or Index
     * @arg Is... Arguments
     * @return Writeable reference to the element
    */
    template<IntOrIndex...T>
    S& subscript(const T... Is) const
    requires((sizeof...(Is) == rank) && Multi<int_t,T...>){
      return mdspan_type::operator()(Is...);
    }

    /** mdspan::operator() is overloaded to accept arrays, and
     * when working with slices it is convenient to pass an
     * array of arguments instead of separate values via some
     * pack-expanding trick. So we also overload MDTensor::operator()
     * to accept arrays.
     * @arg args std::array of subscripts to the tensor
    */
    S& subscript(std::array<int_t,rank> const& args) {
      return mdspan_type::operator()(args);
    }

    FormTensorExpression()

    /** Set all elements of this tensor to a specified scalar
     * @arg rhs Value to which elements will be set
     * @return *this
    */
    auto operator=(S const& rhs){
      auto assignment_lambda = [this,rhs](const Multi<int_t> auto... Is){
        this->subscript(Is...) = rhs;
      };

      BasicDynamicExtentLambda

      Loop<extents_t>()(dynamic_extent_lambda, assignment_lambda);

      return *this;
    }

#if KTENSOR_RANDOM_INIT
    /** Initialize elements of this tensor to random integers betwen -10 and +10
     */
    void initialize_to_random(){
      std::random_device r;
      std::default_random_engine e1(r());
      std::uniform_int_distribution<int> uniform_dist(-10, 10);

      auto assignment_lambda = [this,&e1,&uniform_dist](const Multi<int_t> auto... Is){
        this->subscript(Is...) = (S) uniform_dist(e1);
      };

      BasicDynamicExtentLambda

      Loop<extents_t>()(dynamic_extent_lambda, assignment_lambda);
    }
#endif
};

// Alias for convenient declarations of MDTensors
template<typename S,int_t...D>
using Tensor = MDTensor<S,KTENSOR_MDSPAN_NAMESPACE::extents<int_t,D...>,
                          KTENSOR_MDSPAN_NAMESPACE::layout_right,
                          KTENSOR_MDSPAN_NAMESPACE::default_accessor<S>>;

/**
 * TensorExpressions that wrap MDTensor objects need some additional and some
 * slightly different functionality, so we make a partial specialization of
 * the generic TensorExpression class to those cases.
 * @tparam S Scalar of the underlying MDTensor
 * @tparam X Type representing the extents of the underlying MDTensor
 * @tparam M IndexMap type
 * @tparam I... Index characters of the wrapping TensorExpression
 * @tparam LayoutPolicy Layout policy type for mdspan
 * @tparam AccessorPolicy Accessor policy type for mdspan
 * @note If an MDTensor is subscripted by Indices and ints, the dimensions dims...
 * of the tensor will not be the same set as the dimensions DI... of the
 * TensorExpression wrapping it.
 */
template<typename S, Extent_c X, IndexMap_c M, char...I,
         typename LayoutPolicy, typename AccessorPolicy>
requires (RepeatedIndexDimensionsMatch<M>::value)
class TensorExpression<MDTensor<S, X, LayoutPolicy, AccessorPolicy>, M, I...>{
  private:
    using contents_t = MDTensor<S, X, LayoutPolicy, AccessorPolicy>;
    using expr_t = TensorExpression<contents_t, M, I...>;
    contents_t& contents;
    constexpr static int_t rank = X::rank();
    const std::array<int_t,rank> fixed_args;

  public:
    using return_type = S;
    using map_t = M;

    TensorExpressionConstructor(contents_t&)
    BasicExtentFunction
    BasicContractionCheck
    AllIntegerSubscriptOperator(S&)
    ImplicitSummationFunction
    ReduceToScalar

    /** Assign to the underlying MDTensor from another TensorExpression
     * @tparam B Type wrapped by the TensorExpression on the right-hand side (RHS)
     * @tparam N IndexMap for the RHS
     * @tparam J... Index characters for the RHS expression
     * @arg rhs RHS TensorExpression
     * @return *this
     */
    template<typename B, IndexMap_c N, char...J>
    requires(CommonIndexDimensionsMatch<M,N>::value &&  // Indices which appear on both sides must have the same dimensions
             NonLHSIndicesAreContracting<map_t,TensorExpression<B,N,J...>,J...>::value && // RHS indices must be either on the LHS or contracting
             ContractingIndicesAreFinite<map_t,TensorExpression<B,N,J...>,J...>::value // Contractions on the RHS are over a definite extent
    )
    expr_t& operator=(TensorExpression<B,N,J...> const& rhs)
    {
      auto assignment_lambda = [this,rhs](const Multi<int_t> auto... Is){
        this->subscript(Is...) = rhs.implicit_summation(chars<I...>(), Is...);
      };

      BasicDynamicExtentLambda

      Loop<typename M::extents_t>()(dynamic_extent_lambda, assignment_lambda);

      return *this;
    }

    /** Set all elements of this tensor to a scalar.
     * @arg x Value to set elements to.
     * @arg return *this
     */
    expr_t& operator=(const S& x){
      auto assignment_lambda = [this,x](const Multi<int_t> auto...Is){
        this->subscript(Is...) = x;
      };

      BasicDynamicExtentLambda

      Loop<typename M::extents_t>()(dynamic_extent_lambda, assignment_lambda);

      return *this;
    }

    /*operator return_type() const{
      static_assert(map_t::all_integer,"EXPRESSION IS NOT CONVERTIBLE TO A SCALAR");
      return subscript();
    }*/

    /** Assignment operator for an RHS of identical type.
     * Replaces the operator "=" that gets implicitly deleted due to
     * the specialized TensorExpression having "contents" as a reference member
     * instead of a non-reference member.
     * @arg rhs TensorExpression of the same type as this one
     * @return *this
     */
    expr_t& operator=(const expr_t& rhs){
      return operator=<contents_t>(rhs);
    }
};

} // namespace KTensor

#endif // KTENSOR_EXPRESSION_HPP