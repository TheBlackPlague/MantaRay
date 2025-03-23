//
// Copyright (c) 2023 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_ACCUMULATOROPERATION_H
#define MANTARAY_ACCUMULATOROPERATION_H

namespace MantaRay
{

    /// \brief The accumulator operation.
    /// \details This enum class is used to specify the operation to be performed on the accumulator. These operations
    ///          are used to activate and deactivate elements in the input layers and have them reflected in the
    ///          accumulator.
    enum class AccumulatorOperation
    {

        /// \brief The Accumulator Activation Operation.
        /// \details This operation is used to activate an element in the input layers and have it reflected in the
        ///          accumulator.
        Activate,

        /// \brief The Accumulator Deactivation Operation.
        /// \details This operation is used to deactivate an element in the input layers and have it reflected in the
        ///          accumulator.
        Deactivate

    };

} // MantaRay

#endif //MANTARAY_ACCUMULATOROPERATION_H
