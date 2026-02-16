#include <iostream>
#include <cmath>
#include <ddc/ddc.hpp>
#include "ddc_aliases.hpp"
#include "ddc_alias_inline_functions.hpp"
struct X {
  static constexpr bool PERIODIC = false;
};
struct GridX : UniformGridBase<X> {};

int main(int argc, char** argv){
  Kokkos::ScopeGuard kokkos_scope(argc,argv);
  ddc::ScopeGuard ddc_scope(argc,argv);
  constexpr double x_min = 0.0;
  constexpr double x_max = 1.0;
  constexpr std::size_t num_points = 101;
  ddc::init_discrete_space<GridX>(GridX::init<GridX>(Coord<X>(x_min),Coord<X>(x_max),IdxStep<GridX>(num_points)));
  Idx<GridX> const x_start = ddc::discrete_space<GridX>().front();
  IdxRange<GridX> const grid_x(x_start,IdxStep<GridX>(num_points));
  std::cout<<"  Grid_size : "<<grid_x.size() <<"  points "<<std::endl;
  std::cout<<"  Domain [ "<<x_min<<" , "<<x_max << " ] "<<std::endl;
  DFieldMem<IdxRange<GridX>> temp_alloc(grid_x);
  DFieldMem<IdxRange<GridX>> temp_new_alloc(grid_x);
  DField<IdxRange<GridX>> temp = temp_alloc.span_view();
  DField<IdxRange<GridX>> temp_new = temp_new_alloc.span_view();
  double const sigma = 0.05;
  double const center = 0.5;
  ddc::parallel_for_each(
    Kokkos::DefaultExecutionSpace(),
    grid_x,
    KOKKOS_LAMBDA(Idx<GridX> const idx){
      Coord<X> const x_coord = ddc::coordinate(idx);
      double const x = ddc::get<X>(x_coord);
      std::cout<<" x_coord = "<<x_coord<<" ,               x =  "<<x<<std::endl;
      temp(idx) = std::exp(-0.5 * std::pow((x-center)/sigma,2));
    }
  );
  
  return 0;
}
