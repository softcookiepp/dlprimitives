///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/core/bn.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/gpu/tiered_cache.hpp>

#include <iostream>

namespace dlprim {
namespace core {
	class BatchNormImpl : public BatchNormFwdBwd {
	public:
		virtual ~BatchNormImpl() {}

		BatchNormImpl(Context &ctx,Shape const &s,DataType dtype)
		{
			dt_ = dtype;
			DLPRIM_CHECK(dtype == float_data);
			DLPRIM_CHECK(s.size() >= 2);
			features_ = s[1];
			int total = s.total_size() / features_;
			int second_size = (total + 255) / 256;
			ws_ = features_ * size_of_data_type(dtype) * 5; // two sums + fdy+ fdx+ off
			if(second_size < 64) {
				if(total >= 256)
					wg_ = 256;
				else if(total >= 128)
					wg_ = 128;
				else
					wg_ = 64;
				second_reduce_ = 1;
			}
			else {
				wg_ = 256;
				if(second_size >= 256)
					second_reduce_ = 256;
				else if(second_size >= 128)
					second_reduce_ = 128;
				else
					second_reduce_ = 64;
				ws_ += second_reduce_ * 2 * size_of_data_type(dtype) * features_;
			}
			
			tart::program_ptr sums = gpu::PerDeviceProgramCache::instance().bn_sums(ctx.device(), dtype);
			tart::program_ptr utils = gpu::PerDeviceProgramCache::instance().bn_utils(ctx.device(), dtype);
			if(second_reduce_ > 1)
			{
				sums_ = sums->getKernel("compute_reduce");
				sums_reduce_ = sums->getKernel("reduce");
				dyx_sums_ = sums->getKernel("compute_bwd_reduce");
				dyx_sums_reduce_ = sums->getKernel("reduce_bwd");
			}
			else
			{
				sums_ = sums->getKernel("compute_no_reduce");
				dyx_sums_ = sums->getKernel("compute_bwd_no_reduce");
			}

			update_sums_ = utils->getKernel("update_sums");
			mean_var_to_a_b_ =utils->getKernel("mean_var_to_a_b");
			combine_mean_var_with_gamma_beta_ = utils->getKernel("combine_mean_var_with_gamma_beta");
			compute_backward_factors_ = utils->getKernel("compute_backward_factors");
			forward_ = utils->getKernel("forward");
			backward_data_ = utils->getKernel("backward_data");
			backward_filter_ = utils->getKernel("backward_filter");
			var_gamma_to_a_ = utils->getKernel("var_gamma_to_a");
			backward_test_ = utils->getKernel("backward_test");
		}

		size_t get_plane_size(Shape const &s)
		{
			size_t n=1;
			for(int i=2;i<s.size();i++)
				n*=s[i];
			return n;
		}

		///
		/// Workspace size needed for intermediate results of computations
		///
		virtual size_t workspace()
		{
			return ws_;
		}

		
		virtual void enqueue_calculate_batch_stats(Tensor &x,Tensor &mean,Tensor &var,Tensor &ws,ExecutionContext const &e)
		{
			int batch = x.shape()[0];
			int channels = x.shape()[1];
			DLPRIM_CHECK(channels==features_);
			int hw = get_plane_size(x.shape());
			int p = 0;
			sums_->setArg(p++,batch);
			sums_->setArg(p++,channels);
			sums_->setArg(p++,hw);
			x.set_arg(sums_,p);
			if(second_reduce_ <= 1) {
				mean.set_arg(sums_,p);
				var.set_arg(sums_,p);
				sums_->enqueue({1, features_}, {wg_, second_reduce_});
			}
			else {
				Tensor x_sum = ws.sub_tensor(0,Shape(second_reduce_,features_),dt_);
				Tensor x2_sum = ws.sub_tensor_target_offset(second_reduce_ * features_,
											  Shape(second_reduce_,features_),dt_);
				x_sum.set_arg(sums_,p);
				x2_sum.set_arg(sums_,p);
				p=0;
				sums_reduce_->setArg(p++,features_);
				x_sum.set_arg(sums_reduce_,p);
				x2_sum.set_arg(sums_reduce_,p);
				mean.set_arg(sums_reduce_,p);
				var.set_arg(sums_reduce_,p);
				sums_reduce_->setArg(p++,1.0f/(batch*hw));
				sums_->enqueue({second_reduce_,features_}, {wg_, second_reduce_});
				//e.queue()->sync();
				sums_reduce_->enqueue({1, features_}, {second_reduce_});
			}
		}
		
		virtual void enqueue_update_running_stats(float batch_mean_factor,float running_mean_factor,
												  Tensor &batch_mean,Tensor &running_mean,
												  float batch_var_factor,float running_var_factor,
												  Tensor &batch_var,Tensor &running_var,
												  Tensor &ws,ExecutionContext const &e)
		{
			int p=0;
			update_sums_->setArg(p++,features_);
			batch_mean.set_arg(update_sums_,p);
			batch_var.set_arg(update_sums_,p);
			running_mean.set_arg(update_sums_,p);
			running_var.set_arg(update_sums_,p);
			update_sums_->setArg(p++,batch_mean_factor);
			update_sums_->setArg(p++,running_mean_factor);
			update_sums_->setArg(p++,batch_var_factor);
			update_sums_->setArg(p++,running_var_factor);
			update_sums_->enqueue({features_}, {1, 1, 1});
		}
		void enqueue3D(tart::kernel_ptr &k,int batches,int rc,ExecutionContext const &e,char const *name)
		{
			k->enqueue({rc, features_, batches}, {1, 1, 1});
		}

		///
		/// Peform forward computation as y = (x-mean) / sqrt(var + eps)
		///
		/// Note mean/var can be taken from batch or from global running stats as per user request
		///
		void forward_ab(Tensor &x,Tensor &y,Tensor &a,Tensor &b,ExecutionContext const &e)
		{
			int p = 0;
			int batches = x.shape()[0];
			int rc = get_plane_size(x.shape());
			forward_->setArg(p++,int(x.shape()[0]));
			forward_->setArg(p++,features_);
			forward_->setArg(p++,int(rc));

			x.set_arg(forward_,p);
			y.set_arg(forward_,p);
			a.set_arg(forward_,p);
			b.set_arg(forward_,p);
			enqueue3D(forward_,batches,rc,e,"forward");
		}

		void split_ws_to_a_b_rest(Tensor &ws,Tensor &a,Tensor &b,Tensor &rest)
		{
			split_ws_to_a_b(ws,a,b);

			size_t msize = ws.shape().total_size() * size_of_data_type(ws.dtype());
			size_t items = msize / size_of_data_type(dt_);
			rest = ws.sub_tensor_target_offset(features_ * 2,Shape(items - features_ * 2),dt_);
		}
		void split_ws_to_a_b(Tensor &ws,Tensor &a,Tensor &b)
		{
			a = ws.sub_tensor_target_offset(0,Shape(features_),dt_);
			b = ws.sub_tensor_target_offset(features_,Shape(features_),dt_);
		}

		virtual void enqueue_forward_get_rstd(
											Tensor &x,Tensor &y,
											Tensor &mean,Tensor &var,float eps,Tensor &rstd,
											Tensor &ws,ExecutionContext const &e)
		{
			DLPRIM_CHECK(ws.memory_size() >= ws_);
			Tensor b = ws.sub_tensor_target_offset(0,Shape(features_),dt_);
			int p = 0;
			mean_var_to_a_b_->setArg(p++,features_);
			mean_var_to_a_b_->setArg(p++,eps);

			mean.set_arg(mean_var_to_a_b_,p);
			var.set_arg(mean_var_to_a_b_,p);
			rstd.set_arg(mean_var_to_a_b_,p);
			b.set_arg(mean_var_to_a_b_,p);

			mean_var_to_a_b_->enqueue({features_}, {1, 1, 1});
			forward_ab(x, y, rstd, b, e);

		}
		virtual void enqueue_forward_direct(Tensor &x,Tensor &y,
											Tensor &mean,Tensor &var,float eps,
											Tensor &ws,ExecutionContext const &e)
		{
			DLPRIM_CHECK(ws.memory_size() >= ws_);
			Tensor a,b;
			split_ws_to_a_b(ws,a,b);

			int p = 0;
			mean_var_to_a_b_->setArg(p++,features_);
			mean_var_to_a_b_->setArg(p++,eps);
			mean.set_arg(mean_var_to_a_b_,p);
			var.set_arg(mean_var_to_a_b_,p);
			a.set_arg(mean_var_to_a_b_,p);
			b.set_arg(mean_var_to_a_b_,p);

			mean_var_to_a_b_->enqueue({features_}, {1, 1, 1});
			forward_ab(x, y, a, b, e);

		}
		///
		/// Peform forward computation as y = (x-mean) / sqrt(var + eps) * gamma + beta 
		///
		/// Notes:
		/// - mean/var can be taken from batch or from global running stats as per user request
		/// - mean/var and gamma/beta are converted to single y=ax+b and than computation is done in a single step
		///
		virtual void enqueue_forward_affine(Tensor &x,Tensor &y,
											Tensor &gamma,Tensor &beta,
											Tensor &mean,Tensor &var,
											float eps,
											Tensor &ws,ExecutionContext const &e)
		{
			DLPRIM_CHECK(ws.memory_size() >= ws_);
			Tensor a,b;
			split_ws_to_a_b(ws,a,b);

			int p = 0;
			combine_mean_var_with_gamma_beta_->setArg(p++, features_);
			combine_mean_var_with_gamma_beta_->setArg(p++,eps);
			mean.set_arg(combine_mean_var_with_gamma_beta_,p);
			var.set_arg(combine_mean_var_with_gamma_beta_,p);
			gamma.set_arg(combine_mean_var_with_gamma_beta_,p);
			beta.set_arg(combine_mean_var_with_gamma_beta_,p);
			a.set_arg(combine_mean_var_with_gamma_beta_,p);
			b.set_arg(combine_mean_var_with_gamma_beta_,p);

			combine_mean_var_with_gamma_beta_->enqueue({features_}, {1, 1, 1});
			forward_ab(x, y, a, b, e);
		}

		///
		/// Perform backpropogation calculations
		///
		/// training_mode - assumes that mean/var were calculated on batches of X - they need to be kept from forward stage
		///   otherwise mean/var considered constant values
		///
		/// gamma affine transofrmation after BN
		///
		/// dy - top gradient for backpropogation
		/// dx - calculate backpropogation on X
		/// dgamma - calculate backpropogation gradient for gamma scale
		/// dbeta - calculate backpropogation gradient for beta scale
		/// ws - worksspace 
		///
		virtual void enqueue_backward_affine(bool training_mode,
											 Tensor &x,Tensor &dy,
											 Tensor &mean,Tensor &var,
											 Tensor &gamma,
											 Tensor *dx,float dx_factor,
											 Tensor *dgamma,float dgamma_factor,
											 Tensor *dbeta,float dbeta_factor,
											 float eps,
											 Tensor &ws,ExecutionContext const &e)
		{
			if(!dx && !dgamma && !dbeta)
				return;
			DLPRIM_CHECK(ws.memory_size() >= ws_);
			Tensor dyx_sum,dy_sum,new_ws;
			split_ws_to_a_b_rest(ws,dyx_sum,dy_sum,new_ws);
			int N = 1 + int(dgamma || dbeta) + (dx != nullptr);
			int id = 0;
			calc_sums(x,dy,new_ws,dyx_sum,dy_sum, e);
			if(dgamma || dbeta)
				backward_filter(mean,var,dyx_sum,dy_sum,
								(dgamma ? *dgamma : null_),(dbeta ? *dbeta : null_),
								dgamma_factor,dbeta_factor,eps, e);
			if(dx)
			{
				if(training_mode)
				{
					backward_data_train(x,*dx,dy,mean,var,dy_sum,dyx_sum,gamma,new_ws,eps,dx_factor, e);
				}
				else
				{
					backward_data_test(*dx,dy,var,gamma,new_ws,eps,dx_factor,e);
				}
			}

		}

		void calc_sums(Tensor &x,Tensor &dy,Tensor &ws,Tensor &dyx_sum,Tensor &dy_sum,ExecutionContext const &e)
		{
			int batch = x.shape()[0];
			int channels = x.shape()[1];
			DLPRIM_CHECK(channels==features_);
			int hw = get_plane_size(x.shape());
			int p = 0;
			dyx_sums_->setArg(p++,batch);
			dyx_sums_->setArg(p++,channels);
			dyx_sums_->setArg(p++,hw);
			x.set_arg(dyx_sums_,p);
			dy.set_arg(dyx_sums_,p);
			if(second_reduce_ <= 1) {
				dyx_sum.set_arg(dyx_sums_,p);
				dy_sum.set_arg(dyx_sums_,p);
				dyx_sums_->enqueue({1, features_}, {wg_, second_reduce_});
			}
			else {
				Tensor s1 = ws.sub_tensor(0,Shape(second_reduce_,features_),dt_);
				Tensor s2 = ws.sub_tensor_target_offset(features_ * second_reduce_,
											  Shape(second_reduce_,features_),dt_);
				s1.set_arg(dyx_sums_,p);
				s2.set_arg(dyx_sums_,p);
				p=0;
				dyx_sums_reduce_->setArg(p++,features_);
				s1.set_arg(dyx_sums_reduce_,p);
				s2.set_arg(dyx_sums_reduce_,p);
				dyx_sum.set_arg(dyx_sums_reduce_,p);
				dy_sum.set_arg(dyx_sums_reduce_,p);

				dyx_sums_->enqueue({second_reduce_, features_}, {wg_, second_reduce_});
				dyx_sums_reduce_->enqueue({1, features_}, {second_reduce_});
			}
		}

		void backward_data_test(Tensor &dx,Tensor &dy,Tensor &var,Tensor &gamma,
								Tensor &ws,float eps,float dx_factor,ExecutionContext const &e)
		{
			Tensor  dy_factor = ws.sub_tensor_target_offset(0*features_,Shape(features_),dt_);
			int batches = dx.shape()[0];
			int hw = get_plane_size(dx.shape());
			int p=0;
			var_gamma_to_a_->setArg(p++,features_);
			var_gamma_to_a_->setArg(p++,eps);
			var.set_arg(var_gamma_to_a_,p);
			gamma.set_arg(var_gamma_to_a_,p);
			uint32_t use_gamma = 1;
			var_gamma_to_a_->setArg(p++, use_gamma);
			dy_factor.set_arg(var_gamma_to_a_,p);
			var_gamma_to_a_->enqueue({features_}, {1, 1, 1});
			p=0;
			backward_test_->setArg(p++,batches);
			backward_test_->setArg(p++,features_);
			backward_test_->setArg(p++,hw);
			dx.set_arg(backward_test_,p);
			dy.set_arg(backward_test_,p);
			dy_factor.set_arg(backward_test_,p);
			backward_test_->setArg(p++,dx_factor);
			enqueue3D(backward_test_, batches, hw, e, "backward_data");
		}

		void backward_data_train(Tensor &x,Tensor &dx,Tensor &dy,
								 Tensor &mean,Tensor &var,
								 Tensor &dy_sum,Tensor &dyx_sum,
								 Tensor &gamma,Tensor &ws,
								 float eps,float scale,ExecutionContext const &e)
		{
			Tensor  x_factor = ws.sub_tensor_target_offset(0*features_,Shape(features_),dt_);
			Tensor dy_factor = ws.sub_tensor_target_offset(1*features_,Shape(features_),dt_);
			Tensor  b_offset = ws.sub_tensor_target_offset(2*features_,Shape(features_),dt_);
			int batches = dx.shape()[0];
			int hw = get_plane_size(dx.shape());
			int total = batches*hw;
			int p=0;
			compute_backward_factors_->setArg(p++,features_);
			compute_backward_factors_->setArg(p++,total);
			compute_backward_factors_->setArg(p++,eps);
			mean.set_arg(compute_backward_factors_,p);
			var.set_arg(compute_backward_factors_,p);
			dy_sum.set_arg(compute_backward_factors_,p);
			dyx_sum.set_arg(compute_backward_factors_,p);
			gamma.set_arg(compute_backward_factors_,p);
			uint32_t use_gamma = 1;
			compute_backward_factors_->setArg(p++, use_gamma);
			x_factor.set_arg(compute_backward_factors_,p);
			dy_factor.set_arg(compute_backward_factors_,p);
			b_offset.set_arg(compute_backward_factors_,p);

			compute_backward_factors_->enqueue({features_}, {1, 1, 1});
			p=0;
			backward_data_->setArg(p++,batches);
			backward_data_->setArg(p++,features_);
			backward_data_->setArg(p++,hw);
			x.set_arg(backward_data_,p);
			dy.set_arg(backward_data_,p);
			x_factor.set_arg(backward_data_,p);
			dy_factor.set_arg(backward_data_,p);
			b_offset.set_arg(backward_data_,p);
			dx.set_arg(backward_data_,p);
			backward_data_->setArg(p++,scale);
			enqueue3D(backward_data_, batches, hw, e, "backward_data");
		}

		void backward_filter(Tensor &mean,Tensor &var,Tensor &dyx_sum,Tensor &dy_sum,
							 Tensor &dgamma,Tensor &dbeta,
							 float dg_fact,float db_fact,float eps,ExecutionContext const &e)
		{
			int p=0;
			backward_filter_->setArg(p++,features_);
			mean.set_arg(backward_filter_,p);
			var.set_arg(backward_filter_,p);
			dy_sum.set_arg(backward_filter_,p);
			dyx_sum.set_arg(backward_filter_,p);
			dgamma.set_arg(backward_filter_,p);
			uint32_t use_gamma = 1;
			backward_filter_->setArg(p++, use_gamma);
			dbeta.set_arg(backward_filter_,p);
			uint32_t use_beta = 1; // no idea how to configure whether or not gamma and beta are used. In the kernel it is determined via a nullpointer being there or not..
			backward_filter_->setArg(p++, use_beta);
			backward_filter_->setArg(p++,eps);
			backward_filter_->setArg(p++,dg_fact);
			backward_filter_->setArg(p++,db_fact);
			backward_filter_->enqueue({features_}, {1, 1, 1});
		}

		///
		/// Perform backpropogation calculations for BN without affine addtition Gamma/Beta
		///
		/// training_mode - assumes that mean/var were calculated on batches of X - they need to be kept from forward stage
		///   otherwise mean/var considered constant values
		///
		/// dy - top gradient for backpropogation
		/// dx - calculate backpropogation on X 
		/// ws - worksspace 
		///
		virtual void enqueue_backward_direct(bool training_mode,
											 Tensor &x,Tensor &dy,
											 Tensor &mean,Tensor &var,
											 Tensor &dx,float dx_factor,
											 float eps,
											 Tensor &ws,ExecutionContext const &e)
		{
			DLPRIM_CHECK(ws.memory_size() >= ws_);
			Tensor dyx_sum,dy_sum,new_ws;
			split_ws_to_a_b_rest(ws,dyx_sum,dy_sum,new_ws);
			calc_sums(x,dy,new_ws,dyx_sum,dy_sum,e);
			if(training_mode) {
				backward_data_train(x,dx,dy,mean,var,dy_sum,dyx_sum,null_,new_ws,eps,dx_factor, e);
			}
			else {
				backward_data_test(dx,dy,var,null_,new_ws,eps,dx_factor,e);
			}
		}

		virtual void enqueue_backward_rstd(  Tensor &x,Tensor &dy,
											 Tensor &mean,Tensor &rstd,
											 Tensor &dx,float dx_factor,
											 Tensor &ws,ExecutionContext const &e)
		{
			Tensor dyx_sum,dy_sum,new_ws;
			split_ws_to_a_b_rest(ws,dyx_sum,dy_sum,new_ws);
			calc_sums(x,dy,new_ws,dyx_sum,dy_sum,e);
			backward_data_rstd(x,dx,dy,mean,rstd,dy_sum,dyx_sum,new_ws,dx_factor,e);
		}
		
		void backward_data_rstd(Tensor &x,Tensor &dx,Tensor &dy,
								 Tensor &mean,Tensor &rstd,
								 Tensor &dy_sum,Tensor &dyx_sum,
								 Tensor &ws,
								 float scale,ExecutionContext const &e)
		{   
			Tensor  x_factor = ws.sub_tensor_target_offset(0*features_,Shape(features_),dt_);
			Tensor dy_factor = ws.sub_tensor_target_offset(1*features_,Shape(features_),dt_);
			Tensor  b_offset = ws.sub_tensor_target_offset(2*features_,Shape(features_),dt_);
			int batches = dx.shape()[0];
			int hw = get_plane_size(dx.shape());
			int total = batches*hw;
			int p=0;
			compute_backward_factors_->setArg(p++,features_);
			compute_backward_factors_->setArg(p++,total);
			compute_backward_factors_->setArg(p++,-1.0f); // use rstd instead of var
			mean.set_arg(compute_backward_factors_,p);
			rstd.set_arg(compute_backward_factors_,p);
			dy_sum.set_arg(compute_backward_factors_,p);
			dyx_sum.set_arg(compute_backward_factors_,p);
			
			// use another tensor, who cares
			dyx_sum.set_arg(compute_backward_factors_,p);
			const uint32_t useGamma = 0;
			compute_backward_factors_->setArg(p++, useGamma);
			x_factor.set_arg(compute_backward_factors_,p);
			dy_factor.set_arg(compute_backward_factors_,p);
			b_offset.set_arg(compute_backward_factors_,p);
			compute_backward_factors_->enqueue({features_}, {1, 1, 1});
			p=0;
			backward_data_->setArg(p++,batches);
			backward_data_->setArg(p++,features_);
			backward_data_->setArg(p++,hw);
			x.set_arg(backward_data_,p);
			dy.set_arg(backward_data_,p);
			x_factor.set_arg(backward_data_,p);
			dy_factor.set_arg(backward_data_,p);
			b_offset.set_arg(backward_data_,p);
			dx.set_arg(backward_data_,p);
			backward_data_->setArg(p++,scale);
			enqueue3D(backward_data_, batches, hw, e,"backward_data");
		}

	private:
		int features_;
		size_t ws_;
		int wg_;
		int second_reduce_;
		DataType dt_;
		tart::kernel_ptr sums_,sums_reduce_;
		tart::kernel_ptr dyx_sums_,dyx_sums_reduce_;
		tart::kernel_ptr forward_,backward_data_,backward_filter_;
		tart::kernel_ptr update_sums_;
		tart::kernel_ptr mean_var_to_a_b_;
		tart::kernel_ptr compute_backward_factors_;
		tart::kernel_ptr combine_mean_var_with_gamma_beta_;
		tart::kernel_ptr var_gamma_to_a_;
		tart::kernel_ptr backward_test_;

		Tensor null_;
	};

	std::unique_ptr<BatchNormFwdBwd> BatchNormFwdBwd::create(Context &ctx,Shape const &s,DataType dt)
	{
		std::unique_ptr<BatchNormFwdBwd> r(new BatchNormImpl(ctx,s,dt));
		return r;
	}

} // core_ops
} // dlprim
