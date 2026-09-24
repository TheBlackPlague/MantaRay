//
// Copyright (c) 2026 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_MANTARAY_H
#define MANTARAY_MANTARAY_H

#include "Architecture/Layer.h"
#include "Architecture/Network.h"
#include "Architecture/Quantization.h"
#include "Architecture/Activation/ClippedReLU.h"
#include "Architecture/Activation/Identity.h"
#include "Architecture/Activation/SquaredClippedReLU.h"
#include "Architecture/Topology/Accumulate.h"
#include "Architecture/Topology/Concat.h"
#include "Architecture/Topology/Mirror.h"
#include "Architecture/Topology/Parallel.h"
#include "Architecture/Topology/Residual.h"
#include "Architecture/Topology/Sequence.h"
#include "Architecture/Transform/Affine.h"
#include "IO/BinaryFileStream.h"
#include "IO/BinaryMemoryStream.h"
#include "Runtime/AccumulatorStack.h"
#include "Runtime/Network.h"
#include "Runtime/State.h"

#endif
