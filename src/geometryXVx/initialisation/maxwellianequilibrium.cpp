// SPDX-License-Identifier: MIT

#include <ddc/ddc.hpp>

#include "twostreamequilibrium.hpp"

TwoStreamEquilibrium::TwoStreamEquilibrium(
        host_t<DFieldMemSp> density_eq,
        host_t<DFieldMemSp> temperature_eq,
        host_t<DFieldMemSp> beam_velocity_eq)
    : m_density_eq(std::move(density_eq))
    , m_temperature_eq(std::move(temperature_eq))
    , m_beam_velocity_eq(std::move(beam_velocity_eq))
{
}

DFieldSpVx TwoStreamEquilibrium::operator()(DFieldSpVx const allfequilibrium) const
{
    IdxRangeVx const gridvx = get_idx_range<GridVx>(allfequilibrium);
    IdxRangeSp const gridsp = get_idx_range<Species>(allfequilibrium);

    // Initialisation of the two-stream profile, one species at a time
    DFieldMemVx profile_alloc("two stream (TwoStreamEquilibrium::operator())", gridvx);
    DFieldVx profile = get_field(profile_alloc);
    ddc::host_for_each(gridsp, [&](IdxSp const isp) {
        compute_two_stream(
                profile,
                m_density_eq(isp),
                m_temperature_eq(isp),
                m_beam_velocity_eq(isp));

        const std::source_location location = std::source_location::current();
        ddc::parallel_for_each(
                location.function_name(),
                Kokkos::DefaultExecutionSpace(),
                gridvx,
                KOKKOS_LAMBDA(IdxVx const ivx) { allfequilibrium(isp, ivx) = profile(ivx); });
    });
    return allfequilibrium;
}

void TwoStreamEquilibrium::compute_two_stream(
        DFieldVx const fTwoStream,
        double const density,
        double const temperature,
        double const beam_velocity)
{
    double const inv_sqrt_2piT = 1. / Kokkos::sqrt(2. * M_PI * temperature);
    IdxRangeVx const gridvx = get_idx_range(fTwoStream);
    const std::source_location location = std::source_location::current();
    ddc::parallel_for_each(
            location.function_name(),
            Kokkos::DefaultExecutionSpace(),
            gridvx,
            KOKKOS_LAMBDA(IdxVx const ivx) {
                CoordVx const vx = ddc::coordinate(ivx);
                double const beam_plus
                        = Kokkos::exp(-(vx - beam_velocity) * (vx - beam_velocity)
                                      / (2. * temperature));
                double const beam_minus
                        = Kokkos::exp(-(vx + beam_velocity) * (vx + beam_velocity)
                                      / (2. * temperature));
                fTwoStream(ivx) = 0.5 * density * inv_sqrt_2piT * (beam_plus + beam_minus);
            });
}

TwoStreamEquilibrium twostream_equilibrium::init_from_input(
        IdxRangeSp idx_range_kinsp,
        PC_tree_t const& yaml_input_file)
{
    host_t<DFieldMemSp> density_eq(idx_range_kinsp);
    host_t<DFieldMemSp> temperature_eq(idx_range_kinsp);
    host_t<DFieldMemSp> beam_velocity_eq(idx_range_kinsp);

    for (IdxSp const isp : idx_range_kinsp) {
        PC_tree_t const conf_isp = PCpp_get(yaml_input_file, ".SpeciesInfo[%d]", isp.uid());

        density_eq(isp) = PCpp_double(conf_isp, ".density_eq");
        temperature_eq(isp) = PCpp_double(conf_isp, ".temperature_eq");
        // the beam speed is read from the same key as the Maxwellian's mean velocity
        beam_velocity_eq(isp) = PCpp_double(conf_isp, ".mean_velocity_eq");
    }

    return TwoStreamEquilibrium(
            std::move(density_eq),
            std::move(temperature_eq),
            std::move(beam_velocity_eq));
}
