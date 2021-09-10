// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API DepthToSpaceFusion;
class TRANSFORMATIONS_API DepthToSpaceFusionWithOneTranspose;
class TRANSFORMATIONS_API DepthToSpaceFusionWithMulTransposes;

} // namespace pass
} // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief DepthToSpaceFusion transformation detects Reshape-Transpose-Reshape
 * pattern and tries to fuse it into a single DepthToSpace layer.
 *
 * DepthToSpaceFusion transformation is optional and disabled by default.
 * The transformation can be enabled with callback using setCallback method.
 * See the example below.
 *
 * Callback example:
 *
 *     // This callback enables DepthToSpaceFusion transformation
 *     auto callback = [](const std::shared_ptr<const ngraph::Node> & node) ->
 * bool { return std::dynamic_pointer_cast<const
 * ngraph::opset3::DepthToSpace>(node) != nullptr;
 *     };
 *
 *     auto p = ngraph::pass::DepthToSpaceFusion();
 *     p.setCallback(callback);
 *     p.run_on_function(f);
 *
 */

class ngraph::pass::DepthToSpaceFusionWithOneTranspose
    : public ngraph::pass::MatcherPass {
public:
  NGRAPH_RTTI_DECLARATION;
  DepthToSpaceFusionWithOneTranspose();
};

class ngraph::pass::DepthToSpaceFusionWithMulTransposes
    : public ngraph::pass::MatcherPass {
public:
  NGRAPH_RTTI_DECLARATION;
  DepthToSpaceFusionWithMulTransposes();
};

class ngraph::pass::DepthToSpaceFusion : public ngraph::pass::GraphRewrite {
public:
  NGRAPH_RTTI_DECLARATION;
  DepthToSpaceFusion() {
    add_matcher<ngraph::pass::DepthToSpaceFusionWithOneTranspose>();
    add_matcher<ngraph::pass::DepthToSpaceFusionWithMulTransposes>();
  }
};
