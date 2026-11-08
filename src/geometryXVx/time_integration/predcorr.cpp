
// SPDX-License-Identifier: MIT

#include <cmath>
#include <iostream>

#include <ddc/ddc.hpp>
#include <ddc/pdi.hpp>

#include "fluid_moments.hpp"
#include "iboltzmannsolver.hpp"
#include "iqnsolver.hpp"
#include "neumann_spline_quadrature.hpp"
#include "predcorr.hpp"
#include "quadrature.hpp"




#include "species_info.hpp"

PredCorr::PredCorr(
        IBoltzmannSolver const& boltzmann_solver,
        IQNSolver const& poisson_solver,
        KineticSource const& kinetic_source,
        CollisionsIntra const& collision_intra,
        CollisionsInter const& collision_inter)
    : m_boltzmann_solver(boltzmann_solver)
    , m_poisson_solver(poisson_solver)
    , m_kinetic_source(kinetic_source)
    , m_collision_intra(collision_intra)
    , m_collision_inter(collision_inter)
{
}

DFieldSpXVx PredCorr::operator()(
        DFieldSpXVx const allfdistribu,
        double const time_start,
        double const dt,
        DConstFieldVx quadrature_coefficient,
        DConstFieldX quadrature_coefficient_E,
        int const steps) const
{
    auto allfdistribu_alloc = ddc::create_mirror_view(allfdistribu);
    host_t<DFieldSpXVx> allfdistribu_host = get_field(allfdistribu_alloc);


    Quadrature<IdxRangeVx, IdxRangeSpXVx> const integrate_v(quadrature_coefficient);
    Quadrature<IdxRangeX, IdxRangeX> const integrate_x_1d(quadrature_coefficient_E);
    Quadrature<IdxRangeX, IdxRangeSpX> const integrate_x(quadrature_coefficient_E);
    FluidMoments fluid_moments(integrate_v);

    IdxRangeSpX const idx_range_spx = get_idx_range<Species, GridX>(allfdistribu);
    IdxRangeSp const idx_range_sp = get_idx_range<Species>(allfdistribu);
    IdxRangeX const grid_x = get_idx_range<GridX>(allfdistribu);
    //double const L_x = ddc::coordinate(grid_x.back()) - ddc::coordinate(grid_x.front());

    // 2D moments (Species, GridX)
    DFieldMemSpX density_alloc(idx_range_spx);
    DFieldSpX density = get_field(density_alloc);
    host_t<DFieldMemSpX> density_host_alloc(idx_range_spx);
    host_t<DFieldSpX> density_host = get_field(density_host_alloc);

    DFieldMemSpX mean_velocity_alloc(idx_range_spx);
    DFieldSpX mean_velocity = get_field(mean_velocity_alloc);
    host_t<DFieldMemSpX> mean_velocity_host_alloc(idx_range_spx);
    host_t<DFieldSpX> mean_velocity_host = get_field(mean_velocity_host_alloc);

    DFieldMemSpX temperature_alloc(idx_range_spx);
    DFieldSpX temperature = get_field(temperature_alloc);
    host_t<DFieldMemSpX> temperature_host_alloc(idx_range_spx);
    host_t<DFieldSpX> temperature_host = get_field(temperature_host_alloc);

    // 1D species-averaged moments (Species)
    DFieldMemSp total_density_sp_alloc(idx_range_sp);
    DFieldSp total_density_sp = get_field(total_density_sp_alloc);
    host_t<DFieldMemSp> total_density_sp_host_alloc(idx_range_sp);
    host_t<DFieldSp> total_density_sp_host = get_field(total_density_sp_host_alloc);

    DFieldMemSp avg_velocity_sp_alloc(idx_range_sp);
    DFieldSp avg_velocity_sp = get_field(avg_velocity_sp_alloc);
    host_t<DFieldMemSp> avg_velocity_sp_host_alloc(idx_range_sp);
    host_t<DFieldSp> avg_velocity_sp_host = get_field(avg_velocity_sp_host_alloc);

    DFieldMemSp avg_temperature_sp_alloc(idx_range_sp);
    DFieldSp avg_temperature_sp = get_field(avg_temperature_sp_alloc);
    host_t<DFieldMemSp> avg_temperature_sp_host_alloc(idx_range_sp);
    host_t<DFieldSp> avg_temperature_sp_host = get_field(avg_temperature_sp_host_alloc);

    // Electrostatic potential and electric field
    host_t<DFieldMemX> electrostatic_potential_host(grid_x);
    DFieldMemX electrostatic_potential(grid_x);
    DFieldMemX electric_field(grid_x);
    DFieldMemX electric_field_sq_alloc(grid_x);
    DFieldX electric_field_sq = get_field(electric_field_sq_alloc);

    host_t<DFieldMemSpXVx> allfdistribu_half_t_host(get_idx_range(allfdistribu));
    DFieldMemSpXVx allfdistribu_half_t(get_idx_range(allfdistribu));
    DFieldMemX ones(grid_x);
    ddc::parallel_fill(ones, 1.0);   // adjust to your DDC overload
    double const L_x = integrate_x_1d(Kokkos::DefaultExecutionSpace(), get_const_field(ones));

    auto compute_and_export = [&](char const* const event_name, int const step_iter, double const step_time) {
        // 1. Solve Poisson at current step
        m_poisson_solver(
                get_field(electrostatic_potential),
                get_field(electric_field),
                get_const_field(allfdistribu));

        // 2. Compute int_E2 = \int E^2 dx
        DConstFieldX const electric_field_const = get_const_field(electric_field);
        ddc::parallel_for_each(
                Kokkos::DefaultExecutionSpace(),
                grid_x,
                KOKKOS_LAMBDA(IdxX const ix) {
                    double const e_val = electric_field_const(ix);
                    electric_field_sq(ix) = e_val * e_val;
                });
        double const int_E2 = integrate_x_1d(
                Kokkos::DefaultExecutionSpace(),
                get_const_field(electric_field_sq));

        // 3. Compute 2D fluid moments
        fluid_moments(density, get_const_field(allfdistribu), FluidMoments::MomentDensity {});
        fluid_moments(
                mean_velocity,
                get_const_field(allfdistribu),
                get_const_field(density),
                FluidMoments::MomentVelocity {});
        fluid_moments(
                temperature,
                get_const_field(allfdistribu),
                get_const_field(density),
                get_const_field(mean_velocity),
                FluidMoments::MomentTemperature {});

        // 4. Compute 1D spatial averages over X
        integrate_x(Kokkos::DefaultExecutionSpace(), total_density_sp, get_const_field(density));
        integrate_x(Kokkos::DefaultExecutionSpace(), avg_velocity_sp, get_const_field(mean_velocity));
        integrate_x(Kokkos::DefaultExecutionSpace(), avg_temperature_sp, get_const_field(temperature));

        ddc::parallel_for_each(
                Kokkos::DefaultExecutionSpace(),
                idx_range_sp,
                KOKKOS_LAMBDA(IdxSp const isp) {
                    total_density_sp(isp) /= L_x;
                    avg_velocity_sp(isp) /= L_x;
                    avg_temperature_sp(isp) /= L_x;
                });

        // 5. Deepcopy all device fields to host mirrors
        ddc::parallel_deepcopy(allfdistribu_host, allfdistribu);
        ddc::parallel_deepcopy(electrostatic_potential_host, electrostatic_potential);
        ddc::parallel_deepcopy(density_host, density);
        ddc::parallel_deepcopy(mean_velocity_host, mean_velocity);
        ddc::parallel_deepcopy(temperature_host, temperature);
        ddc::parallel_deepcopy(total_density_sp_host, total_density_sp);
        ddc::parallel_deepcopy(avg_velocity_sp_host, avg_velocity_sp);
        ddc::parallel_deepcopy(avg_temperature_sp_host, avg_temperature_sp);




        // 6. Trigger PDI event (string keys match pdi_out.yml.hpp!)
        Kokkos::Profiling::pushRegion("(GSLX) HDF5_Output");
        ddc::PdiEvent(event_name)
                .with("iter", step_iter)
                .with("time_saved", step_time)
                .with("fdistribu", allfdistribu_host)
                .with("electrostatic_potential", electrostatic_potential_host)
                .with("density", density_host)
                .with("mean_velocity", mean_velocity_host)
                .with("temperature", temperature_host)
                .with("total_density_sp", total_density_sp_host)
                .with("avg_velocity_sp", avg_velocity_sp_host)
                .with("avg_temperature_sp", avg_temperature_sp_host)
                .with("int_E2", int_E2);
        Kokkos::Profiling::popRegion();
    };


    int iter = 0;
    for (; iter < steps; ++iter) {
        Kokkos::Profiling::pushRegion("(GSLX) Time step");
        double const iter_time = time_start + iter * dt;


        compute_and_export("iteration", iter, iter_time);

        // copy fdistribu
        ddc::parallel_deepcopy(allfdistribu_half_t, allfdistribu);

        // predictor
        m_boltzmann_solver(get_field(allfdistribu_half_t), get_const_field(electric_field), dt / 2);

        // computation of the electrostatic potential at time tn+1/2
        // and the associated electric field
        m_poisson_solver(
                get_field(electrostatic_potential),
                get_field(electric_field),
                get_const_field(allfdistribu_half_t));

        // correction on a dt
        m_boltzmann_solver(allfdistribu, get_const_field(electric_field), dt);

        //kinetic_source- term:
        m_kinetic_source(allfdistribu, dt);

        Kokkos::Profiling::popRegion();
    }

    double const final_time = time_start + iter * dt;

    compute_and_export("last_iteration", iter, final_time);


    return allfdistribu;
}
