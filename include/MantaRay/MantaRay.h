//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_MANTARAY_H
#define MANTARAY_MANTARAY_H

#include "Common.h"
#include "Architecture/Architecture.h"
#include "Architecture/FeatureSet.h"
#include "Architecture/FeatureTransformer.h"
#include "Architecture/Output.h"
#include "Architecture/OutputBuckets.h"
#include "Architecture/Signal.h"
#include "Architecture/Tensor.h"
#include "Architecture/Activation/ClippedReLU.h"
#include "Architecture/Activation/ReLU.h"
#include "Architecture/Activation/SquaredClippedReLU.h"
#include "Architecture/Activation/SquaredReLU.h"
#include "Architecture/FeatureSet/Chess768.h"
#include "Architecture/FeatureSet/KingBuckets.h"
#include "Architecture/Topology/Concat.h"
#include "Architecture/Topology/Pairwise.h"
#include "Architecture/Topology/Select.h"
#include "Architecture/Transform/Affine.h"
#include "Architecture/Transform/Schedule.h"
#include "Backend/Facts.h"
#include "Backend/Lowering.h"
#include "Backend/Target.h"
#include "Backend/ISA/Scalar.h"
#include "Backend/Kernel/Dense.h"
#include "Backend/Kernel/Dot.h"
#include "Backend/Kernel/Pack.h"
#include "Backend/Kernel/Pairwise.h"
#include "Backend/Kernel/Reduce.h"
#include "Backend/Kernel/Update.h"
#include "Backend/Stage/DenseAffine.h"
#include "Backend/Stage/Format.h"
#include "Backend/Stage/PairwiseConcat.h"
#include "Backend/Stage/PerspectiveDot.h"
#include "Backend/Stage/Single.h"
#include "IO/Loader.h"
#include "Runtime/AccumulatorStack.h"
#include "Runtime/Delta.h"
#include "Runtime/FinnyTable.h"
#include "Runtime/Network.h"

#endif //MANTARAY_MANTARAY_H
