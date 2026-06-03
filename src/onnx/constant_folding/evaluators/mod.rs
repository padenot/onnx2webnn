/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Tarek Ziadé <tarek@ziade.org>
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Constant evaluators for ONNX operations

mod concat;
mod constant;
mod constant_of_shape;
mod gather;
mod range;
mod reshape_ops;
mod shape;

pub use concat::ConcatEvaluator;
pub use constant::ConstantEvaluator as ConstantOpEvaluator;
pub use constant_of_shape::ConstantOfShapeEvaluator;
pub use gather::GatherEvaluator;
pub use range::RangeEvaluator;
pub use reshape_ops::{CastEvaluator, SqueezeEvaluator, UnsqueezeEvaluator};
pub use shape::ShapeEvaluator;

use crate::onnx::constant_folding::ConstantEvaluator;

/// Get all built-in evaluators
pub fn get_evaluators() -> Vec<Box<dyn ConstantEvaluator>> {
    vec![
        Box::new(ShapeEvaluator),
        Box::new(GatherEvaluator),
        Box::new(ConcatEvaluator),
        Box::new(UnsqueezeEvaluator),
        Box::new(SqueezeEvaluator),
        Box::new(CastEvaluator),
        Box::new(RangeEvaluator),
        Box::new(ConstantOfShapeEvaluator),
        Box::new(ConstantOpEvaluator),
    ]
}
