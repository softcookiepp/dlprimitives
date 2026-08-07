///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/ops/conv2d.hpp>
#include <dlprim/ops/scal.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/gpu/gemm.hpp>
#include <dlprim/utils/json_helpers.hpp>
#include <dlprim/json.hpp>
#include <dlprim/ops/bwd_bias.hpp>
#include <dlprim/ops/initialization.hpp>
#include <dlprim/shared_resource.hpp>
#include <dlprim/ops/activation.hpp>
#include <dlprim/core/common.hpp>
#include <dlprim/core/conv.hpp>
#include <dlprim/core/bias.hpp>
#include <my_cblas.hpp>

namespace dlprim {
   

	Convolution2DConfig Convolution2DConfig::from_json(json::value const &v)
	{
		Convolution2DConfig cfg;
		cfg.channels_in = v.get("channels_in",cfg.channels_in);
		cfg.channels_out = v.get<int>("channels_out");
		utils::get_1dNd_from_json(v,"kernel",cfg.kernel,true);
		utils::get_1dNd_from_json(v,"stride",cfg.stride);
		utils::get_1dNd_from_json(v,"dilate",cfg.dilate);
		utils::get_1dNd_from_json(v,"pad",cfg.pad);
		cfg.groups = v.get("groups",cfg.groups);
		cfg.bias = v.get("bias",cfg.bias);
		cfg.activation = utils::activation_from_json(v); 
		cfg.fwd_algo = v.get("fwd_algo",cfg.fwd_algo);
		cfg.bwd_data_algo = v.get("bwd_data_algo",cfg.bwd_data_algo);
		cfg.bwd_filter_algo = v.get("bwd_filter_algo",cfg.bwd_filter_algo);
		return cfg;
	}
	

	Shape Convolution2D::get_output_shape(Shape const &in)
	{
		DLPRIM_CHECK(in.size() == 4);
		int batch = in[0];
		DLPRIM_CHECK(int(in[1]) == config_.channels_in);
		int ihw[2] = { int(in[2]), int(in[3]) };
		int ohw[2];
		for(int i=0;i<2;i++)        
			ohw[i] = (ihw[i] + 2 * config_.pad[i] - config_.dilate[i] * (config_.kernel[i] - 1) - 1) /  config_.stride[i] + 1;
		DLPRIM_CHECK(ohw[0] > 0);
		DLPRIM_CHECK(ohw[1] > 0);
		return Shape(batch,config_.channels_out,ohw[0],ohw[1]);
	}

	int Convolution2D::get_im2col_width()
	{
		return config_.channels_in / config_.groups * config_.kernel[0] * config_.kernel[1];
	}

	void Convolution2D::initialize_params(std::vector<Tensor> &parameters,ExecutionContext const &e)
	{
		float k = float(config_.groups) / (config_.channels_in * config_.kernel[0] * config_.kernel[1]);
		float range = std::sqrt(k);
		set_to_urandom(parameters.at(0),shared_resource().rng_state(),-range,range,e);
		if(config_.bias)
			set_to_urandom(parameters.at(1),shared_resource().rng_state(),-range,range,e);
	}

	Convolution2D::Convolution2D(Context &ctx,Convolution2DConfig const &cfg) :
		Operator(ctx),
		config_(cfg),
		dtype_(float_data)
	{
		DLPRIM_CHECK(config_.channels_out > 0);
		DLPRIM_CHECK(dtype_==float_data);
		out_h_ = out_w_ = 0;
		in_h_ = in_w_ = 0;
		bs_ = 0;
	}
	
	Convolution2D::~Convolution2D()
	{
	}

	void Convolution2D::setup(std::vector<TensorSpecs> const &in,
							  std::vector<TensorSpecs> &out,
							  std::vector<TensorSpecs> &params,
							  size_t &workspace)
	{
		DLPRIM_CHECK(in.size() == 1);
		Shape in_shape = in[0].shape();
		DLPRIM_CHECK(in_shape.size() == 4);
		int chn   = in_shape[1];
		if(config_.channels_in == -1) {
			config_.channels_in = chn;
		}
		workspace = 0;

		DLPRIM_CHECK(config_.channels_in  % config_.groups == 0);
		DLPRIM_CHECK(config_.channels_out % config_.groups == 0);

		Shape output_shape = get_output_shape(in_shape);
		out.assign({TensorSpecs(output_shape,dtype_)});

		Shape params_shape(config_.channels_out,
						   config_.channels_in / config_.groups,
						   config_.kernel[0],
						   config_.kernel[1]);

		params.push_back(TensorSpecs(params_shape,dtype_));
		if(config_.bias) 
			params.push_back(TensorSpecs(Shape(config_.channels_out),dtype_));

		if(mode_ == CalculationsMode::train) {
			if(config_.bias) {
				bwd_bias_.reset(new BWBias(ctx_,output_shape,dtype_));
			}
			if(config_.activation != StandardActivations::identity)
				activation_ = std::move(Activation::get_bwd_op(ctx_,config_.activation,in[0]));
		}

		in_h_ = in[0].shape()[2];
		in_w_ = in[0].shape()[3];
		out_h_ = output_shape[2];
		out_w_ = output_shape[3];
		bs_ = in[0].shape()[0];


		{
			setup_algo(in_shape);
		}
		
		ws_size_ = calc_workspace(in_shape);
		workspace = ws_size_;
	}
	size_t Convolution2D::calc_workspace(Shape const &in)
	{
		size_t ws = 0;
		{
			if(conv_)
				ws = std::max(ws,conv_->workspace());
			if(conv_bwd_data_)
				ws = std::max(ws,conv_bwd_data_->workspace());
			if(conv_bwd_filter_)
				ws = std::max(ws,conv_bwd_filter_->workspace());
			if(bwd_bias_)
				ws = std::max(ws,bwd_bias_->workspace());
		}
		return ws;
	}
	void Convolution2D::setup_algo(Shape const &in)
	{
		core::Conv2DSettings cfg(config_,in,dtype_);

		conv_ = std::move(core::Conv2DForward::create(
					ctx_,
					cfg,
					config_.bias,
					config_.activation,
					config_.fwd_algo));

		if(mode_ == CalculationsMode::train) {
			conv_bwd_data_ = std::move(core::Conv2DBackwardData::create(
						ctx_,
						cfg,
						config_.bwd_data_algo));

			conv_bwd_filter_ = std::move(core::Conv2DBackwardFilter::create(
						ctx_,
						cfg,
						config_.bwd_filter_algo));

		}
	}

	void Convolution2D::reshape(std::vector<Shape> const &in,
							   std::vector<Shape> &out,
							   size_t &ws)
	{
		DLPRIM_CHECK(in.size() == 1);
		out.assign({get_output_shape(in[0])});
		if(activation_) {
			std::vector<Shape> tmp;
			size_t ws_act =0;
			activation_->reshape(out,tmp,ws_act);
			DLPRIM_CHECK(ws_act == 0);
		}
		if(bwd_bias_) {
			bwd_bias_.reset(new BWBias(ctx_,out[0],dtype_));
		}
		if(in[0][0] > bs_ || in[0][2] != in_h_ || in[0][3] != in_w_) {
			
			setup_algo(in[0]);

			bs_ = in[0][0];
			in_h_ = in[0][2];
			in_w_ = in[0][3];
			out_h_ = out[0][2];
			out_w_ = out[0][3];
		}

		ws = calc_workspace(in[0]);
	}
	void Convolution2D::forward(std::vector<Tensor> &in,std::vector<Tensor> &out,std::vector<Tensor> &parameters,Tensor &ws,
			ExecutionContext const &ectx)
	{
		DLPRIM_CHECK(in.size() == 1);
		DLPRIM_CHECK(out.size() == 1);
		Shape in_shape = in[0].shape();
		Shape out_shape = out[0].shape();
		DLPRIM_CHECK(out_shape == get_output_shape(in_shape));
		DLPRIM_CHECK(parameters.size()==(1u+unsigned(config_.bias)));
		DLPRIM_CHECK(parameters[0].shape() == Shape(config_.channels_out,config_.channels_in / config_.groups,config_.kernel[0],config_.kernel[1]));
		Tensor &W = parameters[0];
		Tensor *bias = nullptr;
		if(config_.bias) {
			DLPRIM_CHECK(parameters[1].shape() == Shape(config_.channels_out)); 
			bias = &parameters[1];
		}

		{
			conv_->enqueue(in[0],W,bias,out[0],ws,0.0f,ectx);
		}
	}

	namespace details {
		
		struct Im2ColOp {
			template<typename DType>
			static void copy(DType &img,DType &im2col)
			{
				im2col = img;
			}
			template<typename DType>
			static void copy_if(DType *img,DType &im2col,bool cond)
			{
				if(cond) {
					im2col = *img;
				}
				else {
					im2col = DType();
				}
			}
			template<typename DType>
			static void pad_zero(DType &im2col)
			{
				im2col = DType();
			}
		};
		struct Col2ImOp {
			template<typename DType>
			static void copy(DType &img,DType &im2col)
			{
				img += im2col;
			}
			template<typename DType>
			static void copy_if(DType *img,DType &im2col,bool cond)
			{
				if(cond) {
					*img += im2col;
				}
			}
			template<typename DType>
			static void pad_zero(DType &)
			{
			}
		};

		template<int K,int S,int P,typename Op,typename Float>
		void im2col_fast(Shape const &in,Shape const &outs,Float *img_in,Float *mat_in)
		{
			int rows = outs[2];
			int cols = outs[3];
			int src_rows = in[2];
			int src_cols = in[3];
			int channels_in = in[1];
			for(int chan = 0;chan < channels_in;chan ++) {
				for(int r=P;r<rows-P;r++) {
					for(int c=P;c<cols-P;c++) {
						int mat_row = r * cols + c;
						int mat_col = chan * (K*K);
						Float *mat = mat_in + mat_row * channels_in * (K*K) + mat_col;
						int y_pos = -P + r * S; 
						int x_pos = -P + c * S;
						Float *img = img_in + src_cols * (chan * src_rows + y_pos) + x_pos;


						for(int dy = 0;dy < K ;dy++, img += src_cols) {
							for(int dx=0;dx < K ;dx++) {
								Op::copy(img[dx],*mat);
								mat++;
							}
						}
					}
				}
				if(P>0) {
					for(int r=0;r<rows;r++) {
						for(int c=0;c<cols;c++) {
							if(c==P && r>=P && r<rows-P) 
								c=cols-P;

							int mat_row = r * cols + c;
							int mat_col = chan * (K*K);
							Float *mat = mat_in + mat_row * channels_in * (K*K) + mat_col;
							int y_pos = -P + r * S;
							int x_pos = -P + c * S;
							Float *img = img_in + src_cols * (chan * src_rows + y_pos) + x_pos;


							for(int dy = 0;dy < K ;dy++, img += src_cols) {
								int y = y_pos + dy;
								if(y >= 0 && y < src_rows) {
									for(int dx=0;dx < K ;dx++) {
										int x = x_pos + dx;
										Op::copy_if(img + dx,*mat,(x >= 0 && x < src_cols));
										mat++;
									}
								}
								else {
									for(int dx=0;dx < K ;dx++) {
										Op::pad_zero(*mat);
										mat++;
									}
								}
							}
						}
					}
				}
			}
		}
	} // details


	template<typename Op,typename DType>
	void Convolution2DBase::im2col(Shape const &in,Shape const &outs,DType *img_in,DType *mat_in,Convolution2DConfig const &config)
	{
		int kern_h   = config.kernel[0];
		int kern_w   = config.kernel[1];
		int pad_h    = config.pad[0];
		int pad_w    = config.pad[1];
		int dilate_h = config.dilate[0];
		int dilate_w = config.dilate[1];
		int stride_h = config.stride[0];
		int stride_w = config.stride[1];

		if(dilate_h == 1 && dilate_h == 1 && kern_h == kern_w && pad_h == pad_w && stride_h == stride_w) {
			int k=kern_w, p = pad_w, s = stride_w;
			if(s < 10 && p < 10) {
				int combine = k*100 + s * 10 + p;
				switch(combine) {
				case 1142: details::im2col_fast<11,4,2,Op>(in,outs,img_in,mat_in); return;
				case  311: details::im2col_fast< 3,1,1,Op>(in,outs,img_in,mat_in); return;
				case  512: details::im2col_fast< 5,1,2,Op>(in,outs,img_in,mat_in); return;
				case  321: details::im2col_fast< 3,2,1,Op>(in,outs,img_in,mat_in); return;
				case  723: details::im2col_fast< 7,2,3,Op>(in,outs,img_in,mat_in); return;
				case  110: details::im2col_fast< 1,1,0,Op>(in,outs,img_in,mat_in); return;
				case  120: details::im2col_fast< 1,2,0,Op>(in,outs,img_in,mat_in); return;
				}
			}
		}

		int rows = outs[2];
		int cols = outs[3];
		int src_rows = in[2];
		int src_cols = in[3];
		int channels_in = in[1];
		for(int chan = 0;chan < channels_in;chan ++) {
			for(int r=0;r<rows;r++) {
				for(int c=0;c<cols;c++) {
					int mat_row = r * cols + c;
					int mat_col = chan * (kern_h * kern_w);
					DType *mat = mat_in + mat_row * channels_in * (kern_h * kern_w) + mat_col;
					int y_pos = -pad_h + r * stride_h;
					int x_pos = -pad_w + c * stride_w;
					DType *img = img_in + src_cols * (chan * src_rows + y_pos) + x_pos;

					for(int dy = 0;dy < kern_h * dilate_h ;dy+= dilate_h, img += src_cols * dilate_h) {
						int y = y_pos + dy;
						if(y >= 0 && y < src_rows) {
							for(int dx=0;dx < kern_w * dilate_w ;dx+= dilate_w) {
								int x = x_pos + dx;
								Op::copy_if(img + dx,*mat,(x >= 0 && x < src_cols));
								mat++;
							}
						}
						else {
							for(int dx=0;dx < kern_w * dilate_w ;dx+= dilate_w) {
								Op::pad_zero(*mat);
								mat++;
							}
						}
					}
				}
			}
		}
	}

	void Convolution2D::backward(std::vector<TensorAndGradient> &input,
								 std::vector<TensorAndGradient> &output,
								 std::vector<TensorAndGradient> &parameters,
								 Tensor &workspace,
								 ExecutionContext const &e)
	{
		int steps =     int(input[0].requires_gradient) 
						+ int(parameters[0].requires_gradient)
						+ int(config_.bias && parameters[1].requires_gradient)
						+ int(config_.activation != StandardActivations::identity);
		int step = 0;
		if(config_.activation != StandardActivations::identity) {
			std::vector<TensorAndGradient> tmp({output[0]}),empty;
			tmp[0].requires_gradient = true;
			tmp[0].accumulate_gradient = 0.0;
			activation_->backward(tmp,tmp,empty,workspace, e);
		}
		if(config_.bias && parameters[1].requires_gradient) {
			DLPRIM_CHECK(bwd_bias_->workspace() <= workspace.shape().total_size());
			bwd_bias_->backward(output[0].diff,
								parameters[1].diff,
								workspace,
								parameters[1].accumulate_gradient,
								e);
		}
		if(parameters[0].requires_gradient) {
			{
				conv_bwd_filter_->enqueue(input[0].data,parameters[0].diff,output[0].diff,
										  workspace,
										  parameters[0].accumulate_gradient,e);
			}
		}

		if(input[0].requires_gradient) {
			{
				auto ec = e;
				conv_bwd_data_->enqueue(input[0].diff,parameters[0].data,output[0].diff,
										workspace,
										input[0].accumulate_gradient, e);
			}
		}
	}




	TransposedConvolution2DConfig TransposedConvolution2DConfig::from_json(json::value const &v)
	{
		Convolution2DConfig src_cfg = Convolution2DConfig::from_json(v);
		TransposedConvolution2DConfig cfg;
		static_cast<Convolution2DConfig&>(cfg) = src_cfg;
		utils::get_1dNd_from_json(v,"output_pad",cfg.output_pad);
		return cfg;
	}

	Shape TransposedConvolution2D::get_output_shape(Shape const &in)
	{
		return core::Conv2DBase::get_output_shape_transposed(config_,in,config_.output_pad);
	}

	int TransposedConvolution2D::get_im2col_width()
	{
		return config_.channels_out / config_.groups * config_.kernel[0] * config_.kernel[1];
	}

	void TransposedConvolution2D::initialize_params(std::vector<Tensor> &parameters,ExecutionContext const &e)
	{
		float k = float(config_.groups) / (config_.channels_out * config_.kernel[0] * config_.kernel[1]);
		float range = std::sqrt(k);
		set_to_urandom(parameters.at(0),shared_resource().rng_state(),-range,range,e);
		if(config_.bias)
			set_to_urandom(parameters.at(1),shared_resource().rng_state(),-range,range,e);
	}

	TransposedConvolution2D::TransposedConvolution2D(Context &ctx,TransposedConvolution2DConfig const &cfg) :
		Operator(ctx),
		config_(cfg),
		dtype_(float_data)
	{
		DLPRIM_CHECK(config_.channels_out > 0);
		DLPRIM_CHECK(dtype_==float_data);
		out_h_ = out_w_ = 0;
		in_h_ = in_w_ = 0;
		bs_ = 0;
		conv_config_ = config_;
		std::swap(conv_config_.channels_out,conv_config_.channels_in);
		conv_config_.bias = false;
		conv_config_.activation = StandardActivations::identity;
	}
	
	TransposedConvolution2D::~TransposedConvolution2D()
	{
	}

	void TransposedConvolution2D::setup(
							  std::vector<TensorSpecs> const &in,
							  std::vector<TensorSpecs> &out,
							  std::vector<TensorSpecs> &params,
							  size_t &workspace)
	{
		DLPRIM_CHECK(in.size() == 1);
		Shape in_shape = in[0].shape();
		DLPRIM_CHECK(in_shape.size() == 4);
		int chn   = in_shape[1];
		if(config_.channels_in == -1) {
			config_.channels_in = chn;
			conv_config_.channels_out = chn;
		}
		workspace = 0;

		DLPRIM_CHECK(config_.channels_in  % config_.groups == 0);
		DLPRIM_CHECK(config_.channels_out % config_.groups == 0);

		Shape output_shape = get_output_shape(in_shape);
		out.assign({TensorSpecs(output_shape,dtype_)});

		Shape params_shape(config_.channels_in,
						   config_.channels_out / config_.groups,
						   config_.kernel[0],
						   config_.kernel[1]);

		params.push_back(TensorSpecs(params_shape,dtype_));
		if(config_.bias) 
			params.push_back(TensorSpecs(Shape(config_.channels_out),dtype_));

		if(mode_ == CalculationsMode::train) {
			if(config_.bias) {
				bwd_bias_.reset(new BWBias(ctx_,output_shape,dtype_));
			}
		}
		if(config_.activation != StandardActivations::identity) {
			ActivationConfig acfg;
			acfg.activation = config_.activation;
			activation_.reset(new Activation(ctx_,acfg));
			std::vector<TensorSpecs> tmp1,tmp2;
			size_t act_ws = 0;
			activation_->mode(mode());
			activation_->setup(out,tmp1,tmp2,act_ws);

			DLPRIM_CHECK(act_ws == 0);
			DLPRIM_CHECK(tmp1 == out);
			DLPRIM_CHECK(tmp2.empty());
		}

		in_h_ = in[0].shape()[2];
		in_w_ = in[0].shape()[3];
		out_h_ = output_shape[2];
		out_w_ = output_shape[3];
		bs_ = in[0].shape()[0];


		{
			setup_algo(in_shape);
		}
		
		ws_size_ = calc_workspace(in_shape);
		workspace = ws_size_;
	}

	size_t TransposedConvolution2D::calc_workspace(Shape const &in)
	{
		size_t ws = 0;
		{
			if(conv_fwd_)
				ws = std::max(ws,conv_fwd_->workspace());
			if(conv_bwd_data_)
				ws = std::max(ws,conv_bwd_data_->workspace());
			if(conv_bwd_filter_)
				ws = std::max(ws,conv_bwd_filter_->workspace());
			if(bwd_bias_)
				ws = std::max(ws,bwd_bias_->workspace());
		}
		return ws;
	}

	void TransposedConvolution2D::setup_algo(Shape const &out)
	{
		Shape in = get_output_shape(out);
		core::Conv2DSettings cfg(conv_config_,in,dtype_);

		conv_fwd_ = std::move(core::Conv2DBackwardData::create(
					ctx_,
					cfg,
					config_.bwd_data_algo));


		if(mode_ == CalculationsMode::train) {
			conv_bwd_data_ = std::move(core::Conv2DForward::create(
						ctx_,
						cfg,
						false, // no bias
						StandardActivations::identity,
						config_.fwd_algo));

			conv_bwd_filter_ = std::move(core::Conv2DBackwardFilter::create(
						ctx_,
						cfg,
						config_.bwd_filter_algo));

		}
	}
	
	void TransposedConvolution2D::reshape(std::vector<Shape> const &in,
							   std::vector<Shape> &out,
							   size_t &ws)
	{
		DLPRIM_CHECK(in.size() == 1);
		out.assign({get_output_shape(in[0])});
		if(activation_) {
			std::vector<Shape> tmp;
			size_t ws_act =0;
			activation_->reshape(out,tmp,ws_act);
			DLPRIM_CHECK(ws_act == 0);
		}
		if(bwd_bias_) {
			bwd_bias_.reset(new BWBias(ctx_,out[0],dtype_));
		}
		if(in[0][0] > bs_ || in[0][2] != in_h_ || in[0][3] != in_w_) {
			
			setup_algo(in[0]);

			bs_ = in[0][0];
			in_h_ = in[0][2];
			in_w_ = in[0][3];
			out_h_ = out[0][2];
			out_w_ = out[0][3];
		}

		ws = calc_workspace(in[0]);
	}

	void TransposedConvolution2D::forward(std::vector<Tensor> &in,std::vector<Tensor> &out,std::vector<Tensor> &parameters,Tensor &ws,
			ExecutionContext const &ectx)
	{
		DLPRIM_CHECK(in.size() == 1);
		DLPRIM_CHECK(out.size() == 1);
		Shape in_shape = in[0].shape();
		Shape out_shape = out[0].shape();
		DLPRIM_CHECK(out_shape == get_output_shape(in_shape));
		DLPRIM_CHECK(parameters.size()==(1u+unsigned(config_.bias)));
		DLPRIM_CHECK(parameters[0].shape() == Shape(config_.channels_in,config_.channels_out / config_.groups,config_.kernel[0],config_.kernel[1]));
		Tensor &W = parameters[0];
		Tensor *bias = nullptr;
		if(config_.bias) {
			DLPRIM_CHECK(parameters[1].shape() == Shape(config_.channels_out)); 
			bias = &parameters[1];
		}

		{
			int total = 1 + bool(bias) + bool(activation_);
			int p = 0;
			conv_fwd_->enqueue(out[0],W,in[0],ws,0.0,ectx);
			if(bias) {
				core::add_bias(out[0],*bias,ectx);
			}
		}
		if(activation_) {
			std::vector<Tensor> dummy;
			activation_->forward(out,out,dummy,ws,ectx);
		}
	}
	void TransposedConvolution2D::backward(
								 std::vector<TensorAndGradient> &input,
								 std::vector<TensorAndGradient> &output,
								 std::vector<TensorAndGradient> &parameters,
								 Tensor &workspace,
								 ExecutionContext const &e)
	{
		int steps =     int(input[0].requires_gradient) 
						+ int(parameters[0].requires_gradient)
						+ int(config_.bias && parameters[1].requires_gradient)
						+ int(config_.activation != StandardActivations::identity);
		int step = 0;
		if(config_.activation != StandardActivations::identity) {
			std::vector<TensorAndGradient> tmp({output[0]}),empty;
			tmp[0].requires_gradient = true;
			tmp[0].accumulate_gradient = 0.0;
			activation_->backward(tmp,tmp,empty,workspace,e);
		}
		if(config_.bias && parameters[1].requires_gradient) {
			DLPRIM_CHECK(bwd_bias_->workspace() <= workspace.shape().total_size());
			bwd_bias_->backward(output[0].diff,
								parameters[1].diff,
								workspace,
								parameters[1].accumulate_gradient,
								e);
		}
		if(parameters[0].requires_gradient) {
			{
				conv_bwd_filter_->enqueue(output[0].diff,parameters[0].diff,input[0].data,
										  workspace,
										  parameters[0].accumulate_gradient,e);
			}
		}

		if(input[0].requires_gradient) {
			{
				conv_bwd_data_->enqueue(output[0].diff,parameters[0].data,nullptr,input[0].diff,
										workspace,
										input[0].accumulate_gradient,e);
			}
		}
	}


} // dlprim
