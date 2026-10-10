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



PredCorr::PredCorr(
        IBoltzmannSolver const& boltzmann_solver,
        IQNSolver const& poisson_solver,
        CollisionsInter const& collisions_inter,
        CollisionsIntra const& collisions_intra,
        KrookSourceConstant const& krook_constant,
        KineticSource const& kinetic_source)
    : m_boltzmann_solver(boltzmann_solver)
    , m_poisson_solver(poisson_solver)
    , m_collisions_intra(collisions_intra)
    , m_collisions_inter(collisions_inter)
    , m_krook_constant(krook_constant)
    , m_kinetic_source(kinetic_source)
{
}

DFieldSpXVx PredCorr::operator()(
        DFieldSpXVx const allfdistribu,
        double const time_start,
        double const dt,
        DConstFieldVx const quadrature_coeffs,
        int const steps) const
{
    auto allfdistribu_alloc = ddc::create_mirror_view(allfdistribu);
    host_t<DFieldSpXVx> allfdistribu_host = get_field(allfdistribu_alloc);
    IdxRangeSpX meshSpX = get_idx_range<Species, GridX>(allfdistribu);
    IdxRangeX meshX = get_idx_range<GridX>(allfdistribu);
    //double const Lx = ddcHelper::total_interval_length(meshX);
    //double const K = 2 * M_PI / Lx ;
    using IdxX = IdxRangeX::discrete_element_type;
    //DFieldMemX E_ext(get_idx_range<GridX>(allfdistribu));
    Quadrature<IdxRangeVx, IdxRangeSpXVx> integrate_v(get_field(quadrature_coeffs));
    FluidMoments flm(integrate_v);
    // electrostatic potential and electric field (depending only on x)
    host_t<DFieldMemX> electrostatic_potential_host(get_idx_range<GridX>(allfdistribu));
    DFieldMemX electrostatic_potential(get_idx_range<GridX>(allfdistribu));
    IdxRangeSpX meshSpx = get_idx_range<Species, GridX>(allfdistribu);
    DFieldMemX electric_field(get_idx_range<GridX>(allfdistribu));
    //host_t<DFieldMemX> electric_field_host(meshX)
    DFieldMemSpX density_alloc(meshSpx);
    DFieldMemSpX mean_velocity_alloc(meshSpx);
    DFieldMemSpX temperature_alloc(meshSpx);
    host_t<DFieldMemSpX> density_host(meshSpx);
    host_t<DFieldMemSpX> mean_velocity_host(meshSpx);
    host_t<DFieldMemSpX> temperature_host(meshSpx);
    //host_t<DFieldMemX> E_ext_host(meshX);
    // a 2D chunk of the same size as fdistribu
    host_t<DFieldMemSpXVx> allfdistribu_half_t_host(get_idx_range(allfdistribu));
    DFieldMemSpXVx allfdistribu_half_t(get_idx_range(allfdistribu));
    m_poisson_solver(
            get_field(electrostatic_potential),
            get_field(electric_field),
            get_const_field(allfdistribu));

    int iter = 0;
    for (; iter < steps; ++iter) {
        Kokkos::Profiling::pushRegion("(GSLX) Time step");
        double const iter_time = time_start + iter * dt;


        // computation of the electrostatic potential at time tn and
        // the associated electric field
        m_poisson_solver(
                get_field(electrostatic_potential),
                get_field(electric_field),
                get_const_field(allfdistribu));
        // copies necessary to PDI
        ddc::parallel_deepcopy(allfdistribu_host, allfdistribu);
        ddc::parallel_deepcopy(electrostatic_potential_host, electrostatic_potential);
        //auto E_ext_host_span = get_field(E_ext_host);
        //ddc::host_for_each(meshX, KOKKOS_LAMBDA(IdxX idx){
        //        CoordX coordx = ddc::coordinate(idx);
        //        E_ext_host_span(idx) = A * cos(K *coordx - w * iter_time);
        //});
        //ddc::parallel_deepcopy(E_ext, E_ext_host);
        flm(get_field(density_alloc),
            get_const_field(allfdistribu),
            FluidMoments::MomentDensity {});

        flm(get_field(mean_velocity_alloc),
            get_const_field(allfdistribu),
            get_const_field(density_alloc),
            FluidMoments::MomentVelocity {});

        flm(get_field(temperature_alloc),
            get_const_field(allfdistribu),
            get_const_field(density_alloc),
            get_const_field(mean_velocity_alloc),
            FluidMoments::MomentTemperature {});

        ddc::parallel_deepcopy(get_field(density_host), get_const_field(density_alloc));
        ddc::parallel_deepcopy(get_field(mean_velocity_host), get_const_field(mean_velocity_alloc));
        ddc::parallel_deepcopy(get_field(temperature_host), get_const_field(temperature_alloc));
        Kokkos::Profiling::pushRegion("(GSLX) HDF5_Output");
        ddc::PdiEvent("iteration")
                .with("iter", iter)
                .with("time_saved", iter_time)
                .with("fdistribu", allfdistribu_host)
                .with("electrostatic_potential", electrostatic_potential_host)
                .with("density", density_host)
                .with("mean_velocity", mean_velocity_host)
                .with("temperature", temperature_host);
        //.with("E_ext", E_ext_host);
        Kokkos::Profiling::popRegion();

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
        //m_boltzmann_solver(get_field(allfdistribu_half_t), get_const_field(E_ext), dt / 2);
        //m_poisson_solver(
        //        get_field(electrostatic_potential),
        //        get_field(E_ext),
        //        get_const_field(allfdistribu_half_t));
        //m_boltzmann_solver(allfdistribu, get_const_field(E_ext), dt);
        m_collisions_intra(allfdistribu, dt);
        m_collisions_inter(allfdistribu, dt);
        m_krook_constant(allfdistribu, dt);
        m_kinetic_source(allfdistribu, dt);
        Kokkos::Profiling::popRegion();
    }

    double const final_time = time_start + iter * dt;
    m_poisson_solver(
            get_field(electrostatic_potential),
            get_field(electric_field),
            get_const_field(allfdistribu));
    //copies necessary to PDI
    ddc::parallel_deepcopy(allfdistribu_host, allfdistribu);
    ddc::parallel_deepcopy(electrostatic_potential_host, electrostatic_potential);
    ddc::parallel_deepcopy(get_field(density_host), get_const_field(density_alloc));
    ddc::parallel_deepcopy(get_field(mean_velocity_host), get_const_field(mean_velocity_alloc));
    ddc::parallel_deepcopy(get_field(temperature_host), get_const_field(temperature_alloc));
    ddc::PdiEvent("last_iteration")
            .with("iter", iter)
            .with("time_saved", final_time)
            .with("fdistribu", allfdistribu_host)
            .with("electrostatic_potential", electrostatic_potential_host)
            .with("density", density_host)
            .with("mean_velocity", mean_velocity_host)
            .with("temperature", temperature_host);
    //.with("E_ext", E_ext_host);


    return allfdistribu;
}
