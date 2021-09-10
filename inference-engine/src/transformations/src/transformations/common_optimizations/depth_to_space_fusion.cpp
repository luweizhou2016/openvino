// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/depth_to_space_fusion.hpp"

#include "itt.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset3.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

bool check_block_first(const ngraph::Shape &shape_input,
                       const ngraph::Shape &shape_reshape_before,
                       const ngraph::AxisVector &permutation,
                       const ngraph::Shape &shape_reshape_after,
                       size_t &possible_block_size) {
  bool is_transformation_valid = true;
  uint64_t spatial_dims = shape_input.size() - 2;
  possible_block_size = shape_reshape_before[1];
  if (possible_block_size == 0)
    return false;
  uint64_t c_dim = static_cast<uint64_t>(
      shape_input[1] / std::pow(possible_block_size, spatial_dims));

  // x' = reshape(data, [N, block_size, block_size, ..., block_size, C /
  // (block_size ^ K), D1, D2, ..., DK])
  ngraph::Shape expected_shape = {shape_input[0]};
  for (uint64_t i = 0; i < spatial_dims; ++i)
    expected_shape.push_back(possible_block_size);
  expected_shape.push_back(c_dim);
  for (uint64_t i = 2; i < shape_input.size(); ++i)
    expected_shape.push_back(shape_input[i]);
  is_transformation_valid &= (expected_shape == shape_reshape_before);

  // x'' = transpose(x', [0,  K + 1,  K + 2, 1, K + 3, 2, K + 4, 3, ..., K + (K
  // + 1), K])
  ngraph::AxisVector expected_permutation = {
      0, static_cast<size_t>(spatial_dims + 1)};
  for (uint64_t i = 2; i < shape_input.size(); ++i) {
    expected_permutation.push_back(spatial_dims + i);
    expected_permutation.push_back(i - 1);
  }
  is_transformation_valid &= (expected_permutation == permutation);

  // y = reshape(x'', [N, C / (block_size ^ K), D1 * block_size, D2 *
  // block_size, D3 * block_size, ..., DK * block_size])
  expected_shape = {shape_input[0], static_cast<size_t>(c_dim)};
  for (uint64_t i = 2; i < shape_input.size(); ++i)
    expected_shape.push_back(shape_input[i] * possible_block_size);
  is_transformation_valid &= (expected_shape == shape_reshape_after);

  return is_transformation_valid;
}

bool check_depth_first(const ngraph::Shape &shape_input,
                       const ngraph::Shape &shape_reshape_before,
                       const ngraph::AxisVector &permutation,
                       const ngraph::Shape &shape_reshape_after,
                       size_t &possible_block_size) {
  bool is_transformation_valid = true;
  uint64_t spatial_dims = shape_input.size() - 2;
  possible_block_size = shape_reshape_before[2];
  if (possible_block_size == 0)
    return false;
  uint64_t c_dim = static_cast<uint64_t>(
      shape_input[1] / std::pow(possible_block_size, spatial_dims));

  // x' = reshape(data, [N, C / (block_size ^ K), block_size, block_size, ...,
  // block_size, D1, D2, ..., DK])
  ngraph::Shape expected_shape = {shape_input[0], static_cast<size_t>(c_dim)};
  for (uint64_t i = 0; i < spatial_dims; ++i)
    expected_shape.push_back(possible_block_size);
  for (uint64_t i = 2; i < shape_input.size(); ++i)
    expected_shape.push_back(shape_input[i]);
  is_transformation_valid &= (expected_shape == shape_reshape_before);

  // x'' = transpose(x', [0,  1,  K + 2, 2, K + 3, 3, K + 4, 4, ..., K + (K +
  // 1), K + 1])
  ngraph::AxisVector expected_permutation = {0, 1};
  for (uint64_t i = 2; i < shape_input.size(); ++i) {
    expected_permutation.push_back(spatial_dims + i);
    expected_permutation.push_back(i);
  }
  is_transformation_valid &= (expected_permutation == permutation);

  // y = reshape(x'', [N, C / (block_size ^ K), D1 * block_size, D2 *
  // block_size, D3 * block_size, ..., DK * block_size])
  expected_shape = {shape_input[0], static_cast<size_t>(c_dim)};
  for (uint64_t i = 2; i < shape_input.size(); ++i)
    expected_shape.push_back(shape_input[i] * possible_block_size);
  is_transformation_valid &= (expected_shape == shape_reshape_after);

  return is_transformation_valid;
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::DepthToSpaceFusion, "DepthToSpaceFusion",
                       0);
NGRAPH_RTTI_DEFINITION(ngraph::pass::DepthToSpaceFusionWithOneTranspose,
                       "DepthToSpaceFusionWithOneTranspose", 0);
NGRAPH_RTTI_DEFINITION(ngraph::pass::DepthToSpaceFusionWithMulTransposes,
                       "DepthToSpaceFusionWithMulTransposes", 0);

ngraph::pass::DepthToSpaceFusionWithOneTranspose::
    DepthToSpaceFusionWithOneTranspose() {
  MATCHER_SCOPE(DepthToSpaceFusionWithOneTranspose);
  auto input0 =
      std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
  auto input1 = std::make_shared<pattern::op::Label>(element::i64, Shape{4});
  auto input2 = std::make_shared<pattern::op::Label>(element::i64, Shape{4});
  auto input3 = std::make_shared<pattern::op::Label>(element::i64, Shape{4});
  auto reshape_before =
      std::make_shared<ngraph::opset3::Reshape>(input0, input1, false);
  auto permute =
      std::make_shared<ngraph::opset3::Transpose>(reshape_before, input2);
  auto reshape_after =
      std::make_shared<ngraph::opset3::Reshape>(permute, input3, false);

  ngraph::matcher_pass_callback callback = [this](pattern::Matcher &m) {
    auto reshape_after =
        std::dynamic_pointer_cast<ngraph::opset3::Reshape>(m.get_match_root());
    if (!reshape_after) {
      return false;
    }

    auto permute = std::dynamic_pointer_cast<ngraph::opset3::Transpose>(
        reshape_after->input_value(0).get_node_shared_ptr());
    if (!permute || permute->get_output_target_inputs(0).size() != 1) {
      return false;
    }

    auto reshape_before = std::dynamic_pointer_cast<ngraph::opset3::Reshape>(
        permute->input_value(0).get_node_shared_ptr());
    if (!reshape_before ||
        reshape_before->get_output_target_inputs(0).size() != 1) {
      return false;
    }

    auto p_shape_input = reshape_before->get_input_partial_shape(0);
    auto p_shape_reshape_before = reshape_before->get_output_partial_shape(0);
    auto p_shape_permute = permute->get_output_partial_shape(0);
    auto p_shape_reshape_after = reshape_after->get_output_partial_shape(0);

    if (p_shape_input.is_dynamic() || p_shape_reshape_before.is_dynamic() ||
        p_shape_permute.is_dynamic() || p_shape_reshape_after.is_dynamic()) {
      return false;
    }

    auto shape_input = p_shape_input.get_shape();
    auto shape_reshape_before = p_shape_reshape_before.get_shape();
    auto shape_permute = p_shape_permute.get_shape();
    auto shape_reshape_after = p_shape_reshape_after.get_shape();

    if (shape_input.size() < 3) {
      return false;
    }

    // input shape: [ batch, C, spatial_dims], expected_shape =
    // spatial_dims.size() * 2 + 2
    size_t expected_shape_size = (shape_input.size() - 2) * 2 + 2;
    if (shape_input.size() != shape_reshape_after.size() ||
        shape_reshape_before.size() != expected_shape_size ||
        shape_permute.size() != expected_shape_size) {
      return false;
    }

    ngraph::AxisVector permutation;
    if (auto input_const = std::dynamic_pointer_cast<opset3::Constant>(
            permute->input_value(1).get_node_shared_ptr())) {
      permutation = input_const->get_axis_vector_val();
    } else {
      return false;
    }

    ngraph::opset3::DepthToSpace::DepthToSpaceMode mode;
    size_t block_size;
    if (check_depth_first(shape_input, shape_reshape_before, permutation,
                          shape_reshape_after, block_size)) {
      mode = ngraph::opset3::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST;
    } else if (check_block_first(shape_input, shape_reshape_before, permutation,
                                 shape_reshape_after, block_size)) {
      mode = ngraph::opset3::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST;
    } else {
      return false;
    }

    auto depth_to_space = std::make_shared<ngraph::opset3::DepthToSpace>(
        reshape_before->input_value(0), mode, block_size);
    depth_to_space->set_friendly_name(reshape_after->get_friendly_name());
    ngraph::copy_runtime_info({reshape_before, permute, reshape_after},
                              depth_to_space);
    ngraph::replace_node(reshape_after, depth_to_space);
    return true;
  };

  auto m =
      std::make_shared<ngraph::pattern::Matcher>(reshape_after, matcher_name);
  register_matcher(m, callback);
}

std::function<bool(const ov::Output<ov::Node>&)>
check_order(const std::vector<size_t> expected_order) {
  return [=](const ov::Output<ov::Node> &value) -> bool {
    auto transpose_node = std::dynamic_pointer_cast<ngraph::opset3::Transpose>(
        value.get_node_shared_ptr());

    if (!transpose_node)
      return false;

    auto input1 = std::dynamic_pointer_cast<ngraph::opset3::Constant>(
        transpose_node->input_value(1).get_node_shared_ptr());

    if (!input1)
      return false;

    auto axis_val = input1->get_axis_vector_val();

    return (std::equal(axis_val.begin(), axis_val.end(), expected_order.begin()) && axis_val.size() == expected_order.size());
  };
}

std::function<bool(const ov::Output<ov::Node> &)>
check_shape(const std::vector<size_t> dims) {
  return [=](const ov::Output<ov::Node> &output) -> bool {
    const auto &shape = output.get_partial_shape();
    return(shape == ov::PartialShape(dims));
  };
}

ngraph::pass::DepthToSpaceFusionWithMulTransposes::
    DepthToSpaceFusionWithMulTransposes() {
  MATCHER_SCOPE(DepthToSpaceFusionWithMulTransposes);
  auto input0_label = std::make_shared<pattern::op::Label>(
      element::f32, Shape{},
      check_shape(std::vector<size_t>{1, 80, 360, 640}));

  auto transpose0_label = ov::pass::pattern::wrap_type<opset3::Transpose>(
      {input0_label, ov::pass::pattern::wrap_type<opset3::Constant>()},
      check_order(std::vector<size_t>{0, 2, 3, 1}));

  auto reshape0_label = ov::pass::pattern::wrap_type<ngraph::opset3::Reshape>(
      {transpose0_label, ov::pass::pattern::wrap_type<opset3::Constant>()},
      check_shape(std::vector<size_t>{1, 360, 640, 20, 2, 2}));

  auto transpose1_label = ov::pass::pattern::wrap_type<opset3::Transpose>(
      {reshape0_label, ov::pass::pattern::wrap_type<opset3::Constant>()},
      check_order(std::vector<size_t>{0, 5, 2, 3, 1, 4}));

  auto reshape1_label = ov::pass::pattern::wrap_type<ngraph::opset3::Reshape>(
      {transpose1_label, ov::pass::pattern::wrap_type<opset3::Constant>()},
      check_shape(std::vector<size_t>{1, 2, 640, 20, 720}));

  auto transpose2_label = ov::pass::pattern::wrap_type<opset3::Transpose>(
      {reshape1_label, ov::pass::pattern::wrap_type<opset3::Constant>()},
      check_order(std::vector<size_t>{0, 2, 1, 3, 4}));

  auto reshape2_label = ov::pass::pattern::wrap_type<ngraph::opset3::Reshape>(
      {transpose2_label, ov::pass::pattern::wrap_type<opset3::Constant>()},
      check_shape(std::vector<size_t>{1, 1280, 20, 720}));

  auto transpose3_label = ov::pass::pattern::wrap_type<opset3::Transpose>(
      {reshape2_label, ov::pass::pattern::wrap_type<opset3::Constant>()},
      check_order(std::vector<size_t>{0, 2, 3, 1}));

  ngraph::matcher_pass_callback callback = [=](pattern::Matcher &m) {
    const auto &pattern_to_output = m.get_pattern_value_map();

    auto transpose0 =
        pattern_to_output.at(transpose0_label).get_node_shared_ptr();
    auto reshape0 = pattern_to_output.at(reshape0_label).get_node_shared_ptr();

    auto transpose1 =
        pattern_to_output.at(transpose1_label).get_node_shared_ptr();
    auto reshape1 = pattern_to_output.at(reshape1_label).get_node_shared_ptr();

    auto transpose2 =
        pattern_to_output.at(transpose2_label).get_node_shared_ptr();
    auto reshape2 = pattern_to_output.at(reshape2_label).get_node_shared_ptr();

    auto transpose3 =
        pattern_to_output.at(transpose3_label).get_node_shared_ptr();

    auto mode = ngraph::opset3::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST;
    std::size_t block_size = 2;

    auto depth_to_space = std::make_shared<ngraph::opset3::DepthToSpace>(
        transpose0->input_value(0), mode, block_size);
    depth_to_space->set_friendly_name(transpose3->get_friendly_name());
    ngraph::copy_runtime_info({transpose0, reshape0, transpose1, reshape1,
                               transpose2, reshape2, transpose3},
                              depth_to_space);
    ngraph::replace_node(transpose3, depth_to_space);
    return true;
  };

  auto m = std::make_shared<ngraph::pattern::Matcher>(transpose3_label,
                                                      matcher_name);
  register_matcher(m, callback);
}
