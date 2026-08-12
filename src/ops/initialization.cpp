///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/ops/initialization.hpp>
#include <dlprim/core/common.hpp>
#include <dlprim/random.hpp>
#include <string.h>
#include <cmath>

namespace dlprim {

void get_seed_seq(size_t total,RandomState &state,RandomState::seed_type &seed,RandomState::sequence_type &seq)
{
    size_t rounds = (total +  philox::result_items - 1) / philox::result_items;
    seed = state.seed();
    seq  = state.sequence_bump(rounds);

}

void set_to_urandom(Tensor &t,RandomState &state,float minv,float maxv,ExecutionContext const &e)
{
    RandomState::seed_type seed;
    RandomState::sequence_type seq;
    get_seed_seq(t.shape().total_size(),state,seed,seq);
    {
        core::fill_random(t,seed,seq,core::rnd_uniform,minv,maxv);
    }
}

void set_to_bernoulli(Tensor &t,RandomState &state,float p,ExecutionContext const &e)
{
    RandomState::seed_type seed;
    RandomState::sequence_type seq;
    get_seed_seq(t.shape().total_size(),state,seed,seq);
    {
        core::fill_random(t,seed,seq,core::rnd_bernoulli,p,0);
    }
}


///
/// set t values to normal distribution with mean and sigma), seed is updated
///
void set_to_normal(Tensor &t,RandomState &state,float mean,float sigma,ExecutionContext const &e)
{
    RandomState::seed_type seed;
    RandomState::sequence_type seq;
    get_seed_seq(t.shape().total_size(),state,seed,seq);
    {
        core::fill_random(t,seed,seq,core::rnd_normal,mean,sigma);
    }
}

} //  dlprim

