// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_attention_gen.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <utility>

#include "../ocl_v2/paged_attention_common.hpp"
#include "intel_gpu/primitives/paged_attention.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "openvino/core/partial_shape.hpp"
#include "paged_attention_inst.h"
#include "primitive_cm_base.hpp"
#include "primitive_inst.h"

#ifdef CM_PA_ENABLE
namespace ov::intel_gpu::cm {

using namespace ov;
using namespace ov::intel_gpu::ocl;
using namespace cldnn;
namespace {

// constexpr ov::element::Type softmax_accumulator_type = ov::element::f32;
// constexpr size_t paged_attention_block_size = 16;
// constexpr size_t seq_len_partition_size = 256;
// constexpr size_t subgroup_size = 16;
constexpr size_t WG_SIZE = 16;
constexpr size_t kv_split_data_size = 16;
constexpr size_t split_output_idx = 3;
// constexpr size_t lse_idx = 4;

}  // namespace

// struct PagedAttentionRuntimeParams : public ImplRuntimeParams {
//     PagedAttentionStage stage;
//     size_t num_of_partitions;
//     size_t partition_size;
//     size_t paged_attention_aligned_seq_len;
//     size_t sdpa_opt_max_seq_len;
//     // size_t sdpa_opt_seq_len_partition_size;
// };

// inline size_t get_target_seq_len_block_size() {
//     constexpr size_t block_size = 16;
//     return block_size;
// }

// inline size_t get_generate_stage_block_size(size_t head_size) {
//     auto preferred_block_size = {4, 2, 1};
//     for (const auto& block_size : preferred_block_size) {
//         if (head_size % (block_size * subgroup_size) == 0) {
//             return block_size;
//         }
//     }
//     return 1;
// }

// This function returns the kv_step and kv_split_len based on the architecture.
// return {kv_step, kv_split_len}
inline std::pair<size_t, size_t> get_kv_split_size(size_t arch) {
    if (arch == 1) {
        return {8, 32};  // For Xe1
    } else if (arch == 2) {
        return {16, 32};  // For Xe2
    }
    OPENVINO_ASSERT(false, "Unsupported architecture for KV split size");
    return {0, 0};  // Fallback case, should not be reached
}

inline size_t get_q_step(size_t arch, bool is_single_token = false) {
    if (arch == 1) {
        return is_single_token ? 1 : 8;  // For Xe1
    } else if (arch == 2) {
        return is_single_token ? 1 : 16;  // For Xe2
    }
    OPENVINO_ASSERT(false, "Unsupported architecture for Q step");
    return 0;  // Fallback case, should not be reached
}

inline size_t get_kv_len(const RuntimeParams& params, const PagedAttentionStage& stage) {
    if (stage == PagedAttentionStage::PREFILL) {
        auto key_shape = params.input_layouts[1].get_shape();
        const size_t kv_len = key_shape[key_shape.size() - 2];
        return kv_len;
    } else if (stage == PagedAttentionStage::GENERATE) {
        // TODO FIX: key_cache shape = [16, 128+4, 4, 2269]
        //  auto key_cache_shape = params.input_layouts[3].get_shape();
        //  const size_t kv_len = key_cache_shape[0] * key_cache_shape[key_cache_shape.size() - 2];
        auto key_shape = params.input_layouts[1].get_shape();
        const size_t kv_len = key_shape[key_shape.size() - 2];
        // size_t i = 0;
        // for (auto& l : params.input_layouts) {
        //     auto _shape = l.get_shape();
        //     std::cout << i++ << " shape: " << _shape.to_string() << std::endl;
        // }
        // std::cout << std::endl;
        return kv_len;
    }
    OPENVINO_ASSERT(false, "Unsupported PagedAttentionStage for get_kv_len");
    return 0;  // Fallback case, should not be reached
}

inline size_t get_split_num(const RuntimeParams& params, const PagedAttentionStage& stage) {
    const size_t kv_len = get_kv_len(params, stage);
    auto xe_arch = params.get_device_info().arch < gpu_arch::xe2 ? 1 : 2;
    const size_t split_num = kv_len / get_kv_split_size(xe_arch).second;

    return split_num;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

JitConstants PASageGeneratorBase::get_jit_constants(const kernel_impl_params& params) const {
    auto jit = KernelGenerator::get_jit_constants(params);
    jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
    // std::cout << "PASageGeneratorBase::get_jit_constants: " << get_entry_point(params) << std::endl;

    auto desc = params.typed_desc<paged_attention>();
    jit.make("HEAD_SIZE", desc->k_head_size);
    jit.make("HEADS_NUM", desc->heads_num);
    jit.make("KV_HEADS_NUM", desc->kv_heads_num);

    const float scale_factor = 1.0 / std::sqrt(static_cast<double>(desc->k_head_size));
    jit.make("SCALE_FACTOR", scale_factor);
    jit.make("CMFLA_SCALE_FACTOR", scale_factor);
    jit.make("CMFLA_NUM_HEADS", desc->heads_num);
    jit.make("CMFLA_HEAD_SIZE", desc->k_head_size);
    jit.make("CMFLA_NUM_KV_HEADS", desc->kv_heads_num);
    //hard code.@todo: using padding to check whether Q,K,V in fused buffer.
    jit.make("CMFLA_QK_FUSED", 0);
    jit.make("CMFLA_V_FUSED", 1);


    jit.make("WG_SIZE_HINT", WG_SIZE);

    auto xe_arch = params.get_device_info().arch < gpu_arch::xe2 ? 1 : 2;
    jit.make("XE_ARCH", xe_arch);

    auto split_size = get_kv_split_size(xe_arch);
    jit.make("KV_STEP", split_size.first);

    jit.make("WG_SIZE", WG_SIZE);
    return jit;
}

Arguments PASageGeneratorKMEAN::get_arguments_desc(const kernel_impl_params& params) const {
    const auto desc = params.typed_desc<paged_attention>();

    OPENVINO_ASSERT(!desc->has_scores_output(), "[GPU][CM] PASageGeneratorKMEAN with scores output is not supported yet");

    Arguments args;
    // self.kernels.enqueue("cm_kmean", kmean_gws, kmean_lws, seq_len, kmean_seq_blk, t_k, t_mean_k[i])

    args.push_back({ArgumentDescriptor::Types::SCALAR, 0});         // seq_len
    args.push_back({ArgumentDescriptor::Types::SCALAR, 1});         // kmean_seq_blk

    args.push_back({ArgumentDescriptor::Types::INPUT, 1});          // key input. input1 in pageattenion primitive.
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 3});// output kmean as internal buffer

    return args;
}

JitConstants PASageGeneratorKMEAN::get_jit_constants(const kernel_impl_params& params) const {
    auto jit = PASageGeneratorBase::get_jit_constants(params);
    const auto desc = params.typed_desc<paged_attention>();

    auto local_sz = params.get_device_info().arch < gpu_arch::xe2 ? 64 : 32;

    jit.make("CMKMEAN_STATE_BLK", CMKMEAN_STATE_BLK);
    jit.make("CMKMEAN_LOCAL_SZ", local_sz);
    jit.make("CMKMEAN_UNROLL_NUM", CMKMEAN_UNROLL_NUM);

    // for (auto& it : jit) {
    //     std::cout << "\tjit[" << it.name << "] = " << it.value << std::endl;
    // }
    // std::cout << std::endl;
    return jit;
}

DispatchDataFunc PASageGeneratorKMEAN::get_dispatch_data_func() const {
    return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
        auto& wgs = kd.params.workGroups;
        auto& scalars = kd.params.scalars;
        auto desc = params.typed_desc<paged_attention>();
        // auto rtp = static_cast<PagedAttentionRuntimeParams*>(rt_params);
        const size_t kv_heads_num = desc->kv_heads_num;
        const size_t head_size = desc->k_head_size;
        const size_t local_sz = params.get_device_info().arch < gpu_arch::xe2 ? 64 : 32;
        auto out_shape = params.output_layouts[0].get_shape();
        const size_t seq_len = out_shape[0];
        size_t kmean_seq_blk = (seq_len + local_sz - 1) / local_sz;
        kmean_seq_blk = align_to(kmean_seq_blk, CMKMEAN_UNROLL_NUM);


        wgs.global = {kv_heads_num, head_size / CMKMEAN_STATE_BLK, local_sz};
        wgs.local = {1, 1, local_sz};

        std::vector<size_t> scaler_value = {seq_len, kmean_seq_blk};
        scalars.resize(scaler_value.size());
        for (size_t i = 0; i < scaler_value.size(); ++i) {
            scalars[i].t = ScalarDescriptor::Types::INT32;
            scalars[i].v.s32 = static_cast<int32_t>(scaler_value[i]);
        }
    }};
}

Arguments PASageGeneratorQuan::get_arguments_desc(const kernel_impl_params& params) const {
    const auto desc = params.typed_desc<paged_attention>();

    OPENVINO_ASSERT(!desc->has_scores_output(), "[GPU][CM] PASageGeneratorQuan with scores output is not supported yet");

    Arguments args;
    //self.kernels.enqueue("cm_quantize_qk", quan_gws, quan_lws, seq_len, t_q, t_k, t_dqscale_q[i], t_dqscale_k[i], t_mean_k[i])

    args.push_back({ArgumentDescriptor::Types::SCALAR, 0});  // seq_len
    args.push_back({ArgumentDescriptor::Types::INPUT, 0});  // query
    args.push_back({ArgumentDescriptor::Types::INPUT, 1});  // key

    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 4}); //dqscale_q
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 5}); //dqscale_k
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 3}); //meanK

    return args;
}

JitConstants PASageGeneratorQuan::get_jit_constants(const kernel_impl_params& params) const {
    auto jit = PASageGeneratorBase::get_jit_constants(params);
    const auto desc = params.typed_desc<paged_attention>();

    // auto xe_arch = params.get_device_info().arch < gpu_arch::xe2 ? 1 : 2;

    // TODO: set causal mask only if needed

    // for (auto& it : jit) {
    //     std::cout << "\tjit[" << it.name << "] = " << it.value << std::endl;
    // }
    // std::cout << std::endl;
    return jit;
}

DispatchDataFunc PASageGeneratorQuan::get_dispatch_data_func() const {
    return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
        auto& wgs = kd.params.workGroups;
        auto& scalars = kd.params.scalars;
        auto desc = params.typed_desc<paged_attention>();
        // auto rtp = static_cast<PagedAttentionRuntimeParams*>(rt_params);
        const size_t kv_heads_num = desc->kv_heads_num;
        // const size_t head_size = desc->k_head_size;
        const size_t local_sz = params.get_device_info().arch < gpu_arch::xe2 ? 64 : 32;
        auto out_shape = params.output_layouts[0].get_shape();
        const size_t seq_len = out_shape[0];

        wgs.global = {align_to(kv_heads_num*seq_len, local_sz)};
        wgs.local = {local_sz};

        std::vector<size_t> scaler_value = {seq_len};
        scalars.resize(scaler_value.size());
        for (size_t i = 0; i < scaler_value.size(); ++i) {
            scalars[i].t = ScalarDescriptor::Types::INT32;
            scalars[i].v.s32 = static_cast<int32_t>(scaler_value[i]);
        }
    }};
}

Arguments PASageGeneratorMultiToken::get_arguments_desc(const kernel_impl_params& params) const {
    const auto desc = params.typed_desc<paged_attention>();

    OPENVINO_ASSERT(!desc->has_scores_output(), "[GPU][CM] PASageGeneratorMultiToken with scores output is not supported yet");

    Arguments args;
    //self.kernels.enqueue("cm_sage_sdpa", GWS, LWS, seq_len, t_q, t_k, t_v, t_dqscale_q[i], t_dqscale_k[i], t_out)

    args.push_back({ArgumentDescriptor::Types::SCALAR, 0});  // seq_len

    args.push_back({ArgumentDescriptor::Types::INPUT, 0});  // quantized query
    args.push_back({ArgumentDescriptor::Types::INPUT, 1});  // qyabtized key
    args.push_back({ArgumentDescriptor::Types::INPUT, 2});  // value

    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 4}); //t_dqscale_q
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 5}); //t_dqscale_k

    args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});


    return args;
}

JitConstants PASageGeneratorMultiToken::get_jit_constants(const kernel_impl_params& params) const {
    auto jit = PASageGeneratorBase::get_jit_constants(params);
    const auto desc = params.typed_desc<paged_attention>();

    // auto xe_arch = params.get_device_info().arch < gpu_arch::xe2 ? 1 : 2;
    // jit.make("Q_STEP", get_q_step(xe_arch, false));

    // TODO: set causal mask only if needed
    auto causal_mask = 1;
    // jit.make("CAUSAL_MASK", causal_mask);
    jit.make("CMFLA_IS_CAUSAL", causal_mask);

    // for (auto& it : jit) {
    //     std::cout << "\tjit[" << it.name << "] = " << it.value << std::endl;
    // }
    // std::cout << std::endl;
    return jit;
}

DispatchDataFunc PASageGeneratorMultiToken::get_dispatch_data_func() const {
    return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
        auto& wgs = kd.params.workGroups;
        auto& scalars = kd.params.scalars;
        auto desc = params.typed_desc<paged_attention>();
        // auto rtp = static_cast<PagedAttentionRuntimeParams*>(rt_params);
        const size_t heads_num = desc->heads_num;
        // const size_t head_size = desc->k_head_size;

        auto out_shape = params.output_layouts[0].get_shape();
        const size_t batch = out_shape.size() < 4 ? 1 : out_shape[0];
        const size_t seq_len = out_shape[0];

        auto xe_arch = params.get_device_info().arch < gpu_arch::xe2 ? 1 : 2;
        const size_t q_step = get_q_step(xe_arch, false);
        const size_t q_group_size = WG_SIZE * q_step;
        const size_t q_threads = align_to(seq_len, q_group_size) / q_step;

        wgs.global = {batch, heads_num, q_threads};
        wgs.local = {1, 1, WG_SIZE};

        // std::cout << "PagedAttentionGeneratorMultiToken::get_dispatch_data_func: "
        //           << "out_shape: " << out_shape.to_string() << ", batch: " << batch << ", heads_num: " << heads_num << ", q_threads: " << q_threads
        //           << ", q_len: " << q_len << ", q_step: " << q_step << std::endl;

        // auto& value_layout = params.input_layouts[2];
        // std::cout << "PagedAttentionGeneratorMultiToken::get_dispatch_data_func: "
        //           << "value_layout: " << value_layout.to_string() << ", v_before_padding: " << v_before_padding << std::endl;
        // Prefill stage: kv_len == q_len
        std::vector<size_t> scaler_value = {seq_len};
        scalars.resize(scaler_value.size());
        for (size_t i = 0; i < scaler_value.size(); ++i) {
            scalars[i].t = ScalarDescriptor::Types::INT32;
            scalars[i].v.s32 = static_cast<int32_t>(scaler_value[i]);
        }
    }};
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

JitConstants PagedAttentionGeneratorBase::get_jit_constants(const kernel_impl_params& params) const {
    auto jit = KernelGenerator::get_jit_constants(params);
    jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
    // std::cout << "PagedAttentionGeneratorBase::get_jit_constants: " << get_entry_point(params) << std::endl;

    auto desc = params.typed_desc<paged_attention>();
    jit.make("HEAD_SIZE", desc->k_head_size);
    jit.make("HEADS_NUM", desc->heads_num);
    jit.make("KV_HEADS_NUM", desc->kv_heads_num);

    const float scale_factor = 1.0 / std::sqrt(static_cast<double>(desc->k_head_size));
    jit.make("SCALE_FACTOR", scale_factor);
    jit.make("CMFLA_SCALE_FACTOR", scale_factor);
    jit.make("CMFLA_NUM_HEADS", desc->heads_num);
    jit.make("CMFLA_HEAD_SIZE", desc->k_head_size);
    jit.make("CMFLA_NUM_KV_HEADS", desc->kv_heads_num);
    jit.make("WG_SIZE_HINT", WG_SIZE);

    auto xe_arch = params.get_device_info().arch < gpu_arch::xe2 ? 1 : 2;
    jit.make("XE_ARCH", xe_arch);

    auto split_size = get_kv_split_size(xe_arch);
    jit.make("KV_STEP", split_size.first);
    jit.make("WG_SIZE", WG_SIZE);
    return jit;
}

Arguments PagedAttentionSDPAGeneratorMultiToken::get_arguments_desc(const kernel_impl_params& params) const {
    const auto desc = params.typed_desc<paged_attention>();

    Arguments args;
    args.push_back({ArgumentDescriptor::Types::INPUT, 0});  // query
    args.push_back({ArgumentDescriptor::Types::INPUT, 1});  // key
    args.push_back({ArgumentDescriptor::Types::INPUT, 2});  // value

    args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

    args.push_back({ArgumentDescriptor::Types::SCALAR, 0});  // q_len
    args.push_back({ArgumentDescriptor::Types::SCALAR, 1});  // kv_len
    args.push_back({ArgumentDescriptor::Types::SCALAR, 2});  // v_before_padding

    return args;
}

JitConstants PagedAttentionSDPAGeneratorMultiToken::get_jit_constants(const kernel_impl_params& params) const {
    auto jit = PagedAttentionGeneratorBase::get_jit_constants(params);
    const auto desc = params.typed_desc<paged_attention>();

    auto xe_arch = params.get_device_info().arch < gpu_arch::xe2 ? 1 : 2;
    jit.make("Q_STEP", get_q_step(xe_arch, false));

    // TODO: set causal mask only if needed
    auto causal_mask = 1;
    jit.make("CAUSAL_MASK", causal_mask);

    // for (auto& it : jit) {
    //     std::cout << "\tjit[" << it.name << "] = " << it.value << std::endl;
    // }
    // std::cout << std::endl;
    return jit;
}

DispatchDataFunc PagedAttentionSDPAGeneratorMultiToken::get_dispatch_data_func() const {
    return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
        auto& wgs = kd.params.workGroups;
        auto& scalars = kd.params.scalars;
        auto desc = params.typed_desc<paged_attention>();
        // auto rtp = static_cast<PagedAttentionRuntimeParams*>(rt_params);
        const size_t heads_num = desc->heads_num;
        // const size_t head_size = desc->k_head_size;

        auto out_shape = params.output_layouts[0].get_shape();
        // const size_t batch = out_shape.size() < 4 ? 1 : out_shape[0];
        const size_t q_len = out_shape[0];

        auto xe_arch = params.get_device_info().arch < gpu_arch::xe2 ? 1 : 2;
        const size_t q_step = get_q_step(xe_arch, false);
        const size_t q_group_size = WG_SIZE * q_step;
        const size_t q_threads = align_to(q_len, q_group_size) / q_step;
        //hardcode batch =1
        wgs.global = {1, heads_num, q_threads};
        wgs.local = {1, 1, WG_SIZE};

        // std::cout << "PagedAttentionGeneratorMultiToken::get_dispatch_data_func: "
        //           << "out_shape: " << out_shape.to_string() << ", batch: " << batch << ", heads_num: " << heads_num << ", q_threads: " << q_threads
        //           << ", q_len: " << q_len << ", q_step: " << q_step << std::endl;

        // auto& value_layout = params.input_layouts[2];
        auto v_before_padding = (desc->kv_heads_num + desc->heads_num) * desc->k_head_size;
        // std::cout << "PagedAttentionGeneratorMultiToken::get_dispatch_data_func: "
        //           << "value_layout: " << value_layout.to_string() << ", v_before_padding: " << v_before_padding << std::endl;
        // Prefill stage: kv_len == q_len
        auto kv_len = q_len;
        std::vector<size_t> scaler_value = {q_len, kv_len, v_before_padding};
        scalars.resize(scaler_value.size());
        for (size_t i = 0; i < scaler_value.size(); ++i) {
            scalars[i].t = ScalarDescriptor::Types::INT32;
            scalars[i].v.s32 = static_cast<int32_t>(scaler_value[i]);
        }
    }};
}

JitConstants PagedAttentionGeneratorSingleToken::get_jit_constants(const kernel_impl_params& params) const {
    auto jit = PagedAttentionGeneratorBase::get_jit_constants(params);
    jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));

    auto xe_arch = params.get_device_info().arch < gpu_arch::xe2 ? 1 : 2;
    jit.make("Q_STEP", get_q_step(xe_arch, true));
    auto kv_split_size = get_kv_split_size(xe_arch);
    jit.make("KV_STEP", kv_split_size.first);
    jit.make("KV_SPLIT_LEN", kv_split_size.second);

    const size_t kv_len = get_kv_len(params, PagedAttentionStage::GENERATE);
    jit.make("KV_LEN", kv_len);

    return jit;
}

Arguments PagedAttentionGeneratorSingleToken::get_arguments_desc(const kernel_impl_params& params) const {
    Arguments args;

    const auto desc = params.typed_desc<paged_attention>();
    // const auto has_scale_input = !desc->scale_val.has_value();
    const auto has_scores_output = params.output_layouts.size() > 1;

    OPENVINO_ASSERT(!has_scores_output, "[GPU][CM] PagedAttentionGeneratorSingleToken with scores output is not supported yet");

    args.push_back({ArgumentDescriptor::Types::INPUT, 0});  // queries
    args.push_back({ArgumentDescriptor::Types::INPUT, 3});  // keys cache
    args.push_back({ArgumentDescriptor::Types::INPUT, 4});  // values cache

    // TODO: HAS_ATTN_MASK_INPUT
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, split_output_idx});      // split output
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, split_output_idx + 1});  // lse output

    args.push_back({ArgumentDescriptor::Types::SCALAR, 0});  // q_len==1
    args.push_back({ArgumentDescriptor::Types::SCALAR, 1});  // kv_len

    return args;
}

DispatchDataFunc PagedAttentionGeneratorSingleToken::get_dispatch_data_func() const {
    return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
        assert(!params.is_dynamic());
        auto& wgs = kd.params.workGroups;
        const auto desc = params.typed_desc<paged_attention>();

        auto rtp = static_cast<PagedAttentionRuntimeParams*>(rt_params);

        const size_t batch = params.input_layouts[0].get_partial_shape()[0].get_length();
        const size_t heads_num = desc->heads_num;
        const size_t split_num = get_split_num(params, rtp->stage);
        wgs.global = {batch, heads_num, split_num};
        wgs.local = {1, 1, WG_SIZE};

        // generate stage: q_len=1, kv_len=past_len + 1
        auto& scalars = kd.params.scalars;
        auto kv_len = rtp->paged_attention_aligned_seq_len;
        std::vector<size_t> scaler_value = {1, kv_len};
        scalars.resize(scaler_value.size());

        // std::cout << "PagedAttentionGeneratorSingleToken::get_dispatch_data_func: "
        //           << "batch: " << batch << ", heads_num: " << heads_num << ", split_num: " << split_num << ", kv_len: " << kv_len << std::endl;

        for (size_t i = 0; i < scaler_value.size(); ++i) {
            scalars[i].t = ScalarDescriptor::Types::INT32;
            scalars[i].v.s32 = static_cast<int32_t>(scaler_value[i]);
        }
    }};
}

JitConstants PagedAttentionGeneratorSingleTokenFinalization::get_jit_constants(const kernel_impl_params& params) const {
    auto jit = PagedAttentionGeneratorBase::get_jit_constants(params);

    const auto desc = params.typed_desc<paged_attention>();
    jit.make("KV_SPLIT_DATA_SIZE", kv_split_data_size);
    auto xe_arch = params.get_device_info().arch < gpu_arch::xe2 ? 1 : 2;
    jit.make("KV_SPLIT_LEN", get_kv_split_size(xe_arch).second);

    // auto key_cache_shape = params.input_layouts[3].get_shape();
    // const size_t kv_len = key_cache_shape[0] * key_cache_shape[key_cache_shape.size() - 2];
    const size_t kv_len = get_kv_len(params, PagedAttentionStage::GENERATE);
    jit.make("KV_LEN", kv_len);

    return jit;
}

Arguments PagedAttentionGeneratorSingleTokenFinalization::get_arguments_desc(const kernel_impl_params& params) const {
    Arguments args;

    args.push_back({ArgumentDescriptor::Types::INPUT, 5});  // past_lens
    args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

    const auto has_scores_output = params.output_layouts.size() > 1;

    OPENVINO_ASSERT(!has_scores_output, "[GPU][CM] PagedAttentionGeneratorSingleTokenFinalization with scores output is not supported yet");

    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, split_output_idx});      // split data
    args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});                              // output
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, split_output_idx + 1});  // values cache

    return args;
}

DispatchDataFunc PagedAttentionGeneratorSingleTokenFinalization::get_dispatch_data_func() const {
    return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
        assert(!params.is_dynamic());
        auto& wgs = kd.params.workGroups;
        auto& scalars = kd.params.scalars;
        scalars.resize(1);

        const auto desc = params.typed_desc<paged_attention>();
        // auto rtp = static_cast<PagedAttentionRuntimeParams*>(rt_params);

        const size_t batch = params.input_layouts[0].get_partial_shape()[0].get_length();
        const size_t heads_num = desc->heads_num;
        const size_t head_size = desc->k_head_size;

        wgs.global = {batch, heads_num, head_size / kv_split_data_size};
        wgs.local = {1, 1, 1};
    }};
}

}  // namespace ov::intel_gpu::cm
#endif