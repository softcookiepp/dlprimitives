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

void set_to_zero(Tensor &t,ExecutionContext const &e)
{
    {
       core::fill_tensor(t,0,e);
    }
}

namespace {
    template<typename T>
    void fill_value(Tensor &t,T v)
    {
        T *p= t.data<T>();
        size_t size = t.shape().total_size();
        for(size_t i=0;i<size;i++)
            p[i] = v;
    }
}

void set_to_constant(Tensor &t,double value,ExecutionContext const &e)
{
    {
        core::fill_tensor(t,value,e);
    }
}

class UrandomConverter {
public:
    UrandomConverter(float min,float max)
    {
        scale_ = (max-min);
        offset_ = min;
    }
    template<typename T>
    void convert(T &vec)
    {
        for(auto &val : vec) {
            val = val * scale_ + offset_;
        }
    }
    float scale_,offset_;
};

class BernoulliConverter {
public:
    BernoulliConverter(float p) : p_(p)
    {
    }
    template<typename T>
    void convert(T &vec)
    {
        for(auto &val : vec) {
            val = val < p_ ? 1 : 0;
        }
    }
    float p_;
};




class NormalConverter {
public:
    NormalConverter(float mu,float sigma)
    {
        mu_ = mu;
        sigma_ = sigma;
    }
    void convert(philox::float_result_type &f)
    {
        convert_pair(f[0],f[1]);
        convert_pair(f[2],f[3]);
    }
private:
    void convert_pair(float &r1,float &r2)
    {
        float scale = std::sqrt(-2.0f*std::log(1.0f - r1)) * sigma_;
        ///                                     r1 in [0, 1)
        float angle = (2.0f*3.1415926535f)*r2;
        r1 = scale*std::cos(angle) + mu_;
        r2 = scale*std::sin(angle) + mu_;
    }
    float mu_,sigma_;
};

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
        core::fill_random(t,seed,seq,core::rnd_uniform,minv,maxv,e);
    }
}

void set_to_bernoulli(Tensor &t,RandomState &state,float p,ExecutionContext const &e)
{
    RandomState::seed_type seed;
    RandomState::sequence_type seq;
    get_seed_seq(t.shape().total_size(),state,seed,seq);
    {
        core::fill_random(t,seed,seq,core::rnd_bernoulli,p,0,e);
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
        core::fill_random(t,seed,seq,core::rnd_normal,mean,sigma,e);
    }
}

} //  dlprim

