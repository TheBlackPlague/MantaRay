//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_REFERENCE_INTERPRETER_H
#define MANTARAY_REFERENCE_INTERPRETER_H

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <vector>

#include "../Common.h"
#include "../FeatureSet/FeatureSet.h"
#include "../IO/Loader.h"
#include "../IR/Analysis.h"
#include "../IR/GraphIR.h"

namespace MantaRay::Reference
{

    struct Evaluation
    {

        std::vector<i32> Heads      {};
        std::vector<f64> Raw        {};
        u64              Violations = 0;

    };

    namespace Detail
    {

        constexpr i64 WrapTo(const i64 value, const IR::DType type)
        {
            switch (type) {
                case IR::DType::I08: return static_cast<i08>(static_cast<u08>(value));
                case IR::DType::U08: return static_cast<u08>(value);
                case IR::DType::I16: return static_cast<i16>(static_cast<u16>(value));
                case IR::DType::U16: return static_cast<u16>(value);
                case IR::DType::I32: return static_cast<i32>(static_cast<u32>(value));
                default            : return value;
            }
        }

        template<typename W>
        f32 Reduce(const IR::Reduction order, const std::span<const f32> x, const std::span<const W> w, const f32 bias)
        {
            const u32 count = static_cast<u32>(w.size());

            if (order == IR::Reduction::BiasFirstFma) {
                f32 sum = bias;

                for (u32 k = 0; k < count; k++) sum = std::fma(x[k], static_cast<f32>(w[k]), sum);

                return sum;
            }

            if (order == IR::Reduction::Lanes16) {
                Array<f32, 16> lanes {};

                for (u32 k = 0; k < count; k++) lanes[k % 16] = std::fma(x[k], static_cast<f32>(w[k]), lanes[k % 16]);

                for (u32 width = 8; width > 0    ; width /= 2)
                for (u32 i     = 0; i     < width; i++       ) lanes[i] += lanes[i + width];

                return lanes[0];
            }

            f32 sum = 0;

            for (u32 k = 0; k < count; k++) {
                const f32 product = x[k] * static_cast<f32>(w[k]);

                sum += product;
            }

            return sum;
        }

        inline f32 Factor(const Rational value)
        { return static_cast<f32>(value.Numerator) / static_cast<f32>(value.Denominator); }

        inline f32 Real(const Rational value) { return static_cast<f32>(ToDouble(value)); }

    }

    template<typename M>
    class Interpreter
    {

        constexpr static auto& Graph = M::Graph;

        struct Values
        {

            std::vector<std::vector<i64>> Integers = std::vector<std::vector<i64>>(M::Nodes);
            std::vector<std::vector<f32>> Floats   = std::vector<std::vector<f32>>(M::Nodes);
            std::vector<std::vector<u32>> Features = std::vector<std::vector<u32>>(M::Nodes);

        };

        struct Source
        {

            s00 Node  = 0;
            u64 Index = 0;

        };

        template<typename Board>
        static std::vector<u32> Active(const u16 binding, const u08 perspective, const Board& board)
        {
            std::vector<u32> active {};

            [&]<typename... All>(List<All...>) {
                u16 index = 0;

                ([&] {
                    if (index++ != binding) return;

                    if constexpr (FeatureSet<All>)
                        All::Active(perspective, board, [&](const u32 feature) { active.push_back(feature); });
                }(), ...);
            }(typename M::Bindings {});

            return active;
        }

        template<typename Board>
        static u32 Bucket(const u16 binding, const Board& board)
        {
            u32 bucket = 0;

            [&]<typename... All>(List<All...>) {
                u16 index = 0;

                ([&] {
                    if (index++ != binding) return;

                    if constexpr (Selector<All>) bucket = All::Of(board);
                }(), ...);
            }(typename M::Bindings {});

            return bucket;
        }

        static bool Integer(const s00 node) { return M::Facts[node].Domain == IR::Domain::Integer; }

        template<typename T>
        static void Set(Values& values, const s00 node, const u32 k, const T value)
        {
            if (Integer(node)) values.Integers[node][k] = static_cast<i64>(value);
            else               values.Floats  [node][k] = static_cast<f32>(value);
        }

        template<typename T>
        static void Accumulate(Values& values, const s00 node, const u32 k, const T value)
        {
            if (Integer(node)) values.Integers[node][k] += static_cast<i64>(value);
            else               values.Floats  [node][k] += static_cast<f32>(value);
        }

        static void Copy(Values& values, const s00 node, const u32 k, const Source source)
        {
            if (Integer(node)) values.Integers[node][k] = values.Integers[source.Node][source.Index];
            else               values.Floats  [node][k] = values.Floats  [source.Node][source.Index];
        }

        template<typename Board>
        static void Collect(const Board& board, Values& values, const s00 i)
        {
            const IR::Node& node        = Graph.Nodes[i];
            const u08       stm         = board.SideToMove();
            const u08       perspective = node.Perspective == IR::Perspective::Ntm ? static_cast<u08>(stm ^ 1) : stm;

            values.Features[i] = Active(static_cast<u16>(node.Binding - 1), perspective, board);
        }

        static void Parameter(const IO::Tensors<M>& tensors, Values& values, const s00 i)
        {
            const IR::Node& node = Graph.Nodes[i];

            IO::Detail::Visit(tensors, static_cast<u16>(node.Tensor - 1), [&](const auto weights) {
                for (u32 o = node.Begin; o < node.End; o++) Set(values, i, o - node.Begin, weights[o]);
            });
        }

        static void Sparse(const IO::Tensors<M>& tensors, Values& values, const s00 i)
        {
            const IR::Node& node    = Graph.Nodes[i];
            const u32       columns = Graph.Tensors[node.Tensor - 1].Columns;

            IO::Detail::Visit(tensors, static_cast<u16>(node.Tensor - 1), [&](const auto weights) {
                for (const u32 feature : values.Features[node.Inputs[0]])
                for (u32 o = node.Begin; o < node.End; o++)
                    Accumulate(values, i, o - node.Begin, weights[u64 { o } * columns + node.Offset + feature]);
            });
        }

        static void Matmul(const IO::Tensors<M>& tensors, Values& values, const s00 i)
        {
            const IR::Node& node    = Graph.Nodes[i];
            const u32       columns = Graph.Tensors[node.Tensor - 1].Columns;
            const s00       a       = node.Inputs[0];
            const s00       b       = node.Inputs[1];
            const bool      real    = !Integer(i);

            IO::Detail::Visit(tensors, static_cast<u16>(node.Tensor - 1), [&](const auto weights) {
                for (u32 o = node.Begin; o < node.End; o++) {
                    const u32  k    = o - node.Begin;
                    const auto row  = weights.subspan(u64 { o } * columns, columns);
                    const f32  bias = real && node.InputCount == 2 ? values.Floats[b][k] : 0.0F;

                    if (real) {
                        values.Floats[i][k] = Detail::Reduce(node.Reduction, values.Floats[a], row, bias);

                        continue;
                    }

                    for (u32 c = 0; c < columns; c++)
                        values.Integers[i][k] += values.Integers[a][c] * static_cast<i64>(row[c]);
                }
            });
        }

        template<typename T>
        static T Combine(const IR::Op operation, const T x, const T y)
        {
            switch (operation) {
                case IR::Op::Add: return x + y;
                case IR::Op::Sub: return x - y;
                case IR::Op::Mul: return x * y;
                case IR::Op::Min: return std::min(x, y);
                case IR::Op::Max: return std::max(x, y);
                default         : return static_cast<T>(std::pow(x, y));
            }
        }

        static void Elementwise(Values& values, const s00 i)
        {
            const IR::Node& node     = Graph.Nodes[i];
            const s00       a        = node.Inputs[0];
            const s00       b        = node.Inputs[1];
            const Rational  constant = Graph.Nodes[b].Low;
            const bool      scalar   = M::Facts[b].Domain == IR::Domain::Constant;
            const bool      multiply = node.Operation == IR::Op::Mul;

            const bool integral = scalar && Integer(i);

            const i64 integer = !integral ? 0 : multiply ? constant.Numerator : Floor(constant * M::Facts[a].Scale);
            const f32 real    = scalar ? Detail::Real(constant) : 0.0F;

            const auto integerAt = [&](const u32 k) { return scalar ? integer : values.Integers[b][k]; };
            const auto realAt    = [&](const u32 k) { return scalar ? real    : values.Floats  [b][k]; };

            for (u32 k = 0; k < M::Facts[i].Size; k++)
                if (Integer(i)) values.Integers[i][k] = Combine(node.Operation, values.Integers[a][k], integerAt(k));
                else            values.Floats  [i][k] = Combine(node.Operation, values.Floats  [a][k], realAt   (k));
        }

        static void Clamp(Values& values, const s00 i)
        {
            const IR::Node& node  = Graph.Nodes[i];
            const s00       a     = node.Inputs[0];
            const Rational  scale = M::Facts[a].Scale;

            const bool low  = node.Bounds == IR::Bound::Low  || node.Bounds == IR::Bound::Both;
            const bool high = node.Bounds == IR::Bound::High || node.Bounds == IR::Bound::Both;

            const i64 floor   = low  ? Floor(node.Low  * scale) : std::numeric_limits<i64>::min();
            const i64 ceiling = high ? Floor(node.High * scale) : std::numeric_limits<i64>::max();
            const f32 lower   = low  ? Detail::Real(node.Low )  : -std::numeric_limits<f32>::infinity();
            const f32 upper   = high ? Detail::Real(node.High)  :  std::numeric_limits<f32>::infinity();

            for (u32 k = 0; k < M::Facts[i].Size; k++)
                if (Integer(i)) values.Integers[i][k] = std::min(std::max(values.Integers[a][k], floor), ceiling);
                else            values.Floats  [i][k] = std::min(std::max(values.Floats  [a][k], lower), upper  );
        }

        static f32 Apply(const IR::Function function, const f32 x)
        {
            switch (function) {
                case IR::Function::Abs    : return std::fabs(x);
                case IR::Function::Exp    : return std::exp(x);
                case IR::Function::Sigmoid: return 1.0F / (1.0F + std::exp(-x));
                case IR::Function::Tanh   : return std::tanh(x);
                default                   : return x;
            }
        }

        static void Unary(Values& values, const s00 i)
        {
            const IR::Node& node = Graph.Nodes[i];
            const s00       a    = node.Inputs[0];

            for (u32 k = 0; k < M::Facts[i].Size; k++)
                if (Integer(i)) values.Integers[i][k] = std::abs(values.Integers[a][k]);
                else            values.Floats  [i][k] = Apply(node.Function, values.Floats[a][k]);
        }

        static i64 Shifted(const IR::Node& node, const i64 x)
        {
            switch (node.Operation) {
                case IR::Op::ShiftRight: return x >> node.Shift;
                case IR::Op::ShiftLeft : return x << node.Shift;
                case IR::Op::Divide    : return x /  node.Low.Numerator;
                case IR::Op::Wrap      : return Detail::WrapTo(x, node.Type);
                default                : return x;
            }
        }

        static void Shift(Values& values, const s00 i)
        {
            const IR::Node& node = Graph.Nodes[i];
            const s00       a    = node.Inputs[0];

            for (u32 k = 0; k < M::Facts[i].Size; k++) values.Integers[i][k] = Shifted(node, values.Integers[a][k]);
        }

        static void Dequantize(Values& values, const s00 i)
        {
            const IR::Node& node  = Graph.Nodes[i];
            const s00       a     = node.Inputs[0];
            const s00       b     = node.Inputs[1];
            const Rational  scale = M::Facts[a].Scale;

            const f32 factor = Defined(node.Low) ? Detail::Factor(node.Low)
                                                 : Detail::Factor(Ratio(scale.Denominator, scale.Numerator));

            for (u32 k = 0; k < M::Facts[i].Size; k++) {
                const f32 x = static_cast<f32>(values.Integers[a][k]);

                values.Floats[i][k] = node.InputCount == 2 ? std::fma(x, factor, values.Floats[b][k]) : x * factor;
            }
        }

        static Source SourceOf(const IR::Node& node, const u32 k, const u32 bucket, const u32 size)
        {
            const u32 left = M::Facts[node.Inputs[0]].Size;

            switch (node.Operation) {
                case IR::Op::Slice : return { .Node = node.Inputs[0], .Index = u64 { node.Begin } + k   };
                case IR::Op::Select: return { .Node = node.Inputs[0], .Index = u64 { bucket } * size + k };
                default            : return k < left ? Source { .Node = node.Inputs[0], .Index = k        }
                                                     : Source { .Node = node.Inputs[1], .Index = k - left };
            }
        }

        template<typename Board>
        static void Gather(const Board& board, Values& values, const s00 i)
        {
            const IR::Node& node   = Graph.Nodes[i];
            const u32       size   = M::Facts[i].Size;
            const bool      select = node.Operation == IR::Op::Select;
            const u32       bucket = select ? Bucket(static_cast<u16>(node.Binding - 1), board) : 0;

            for (u32 k = 0; k < size; k++) Copy(values, i, k, SourceOf(node, k, bucket, size));
        }

        static void Sum(Values& values, const s00 i)
        {
            const s00 a = Graph.Nodes[i].Inputs[0];

            for (u32 k = 0; k < M::Facts[a].Size; k++)
                if (Integer(i)) values.Integers[i][0] += values.Integers[a][k];
                else            values.Floats  [i][0] += values.Floats  [a][k];
        }

        template<typename Board>
        static void Step(const IO::Tensors<M>& tensors, const Board& board, Values& values, const s00 i)
        {
            const u32 size = M::Facts[i].Size;

            values.Integers[i].resize(Integer(i) ? size : 0);
            values.Floats  [i].resize(Integer(i) ? 0    : size);

            switch (Graph.Nodes[i].Operation) {
                case IR::Op::Input       : Collect    (board  , values, i); break;
                case IR::Op::Parameter   : Parameter  (tensors, values, i); break;
                case IR::Op::SparseMatmul: Sparse     (tensors, values, i); break;
                case IR::Op::Matmul      : Matmul     (tensors, values, i); break;
                case IR::Op::Add         :
                case IR::Op::Sub         :
                case IR::Op::Mul         :
                case IR::Op::Min         :
                case IR::Op::Max         :
                case IR::Op::Pow         : Elementwise(         values, i); break;
                case IR::Op::Clamp       : Clamp      (         values, i); break;
                case IR::Op::Unary       : Unary      (         values, i); break;
                case IR::Op::ShiftRight  :
                case IR::Op::ShiftLeft   :
                case IR::Op::Divide      :
                case IR::Op::Wrap        :
                case IR::Op::Verify      :
                case IR::Op::Assume      : Shift      (         values, i); break;
                case IR::Op::Dequantize  : Dequantize (         values, i); break;
                case IR::Op::Slice       :
                case IR::Op::Concat      :
                case IR::Op::Select      : Gather     (board  , values, i); break;
                case IR::Op::Sum         : Sum        (         values, i); break;
                default                  : break;
            }
        }

        static u64 Violations(const Values& values, const s00 i)
        {
            const IR::Fact& fact = M::Facts[i];

            if (!Integer(i) || fact.Provenance != IR::Provenance::Proven) return 0;

            const auto outside = [&](const i64 value) { return value < fact.Range.Min || value > fact.Range.Max; };

            return static_cast<u64>(std::ranges::count_if(values.Integers[i], outside));
        }

        static void Head(const Values& values, const u16 h, Evaluation& result)
        {
            using Wide = MantaRay::Detail::i128;

            const IR::Head& head = Graph.Heads[h];
            const IR::Fact& fact = M::Facts[head.Node];

            if (fact.Domain != IR::Domain::Integer) {
                for (const f32 value : values.Floats[head.Node]) {
                    const f32 scaled = value * Detail::Real(head.Scale);

                    result.Heads.push_back(static_cast<i32>(std::clamp(scaled, -2147483648.0F, 2147483520.0F)));
                    result.Raw  .push_back(value);
                }

                return;
            }

            const Wide numerator   = Wide { head.Scale.Numerator   } * fact.Scale.Denominator;
            const Wide denominator = Wide { head.Scale.Denominator } * fact.Scale.Numerator  ;

            for (const i64 value : values.Integers[head.Node]) {
                result.Heads.push_back(static_cast<i32>(Wide { value } * numerator / denominator));
                result.Raw  .push_back(static_cast<f64>(value) / ToDouble(fact.Scale));
            }
        }

        public:
        template<typename Board>
        static Evaluation Evaluate(const IO::Tensors<M>& tensors, const Board& board)
        {
            Values     values {};
            Evaluation result {};

            for (s00 i = 0; i < M::Nodes; i++) {
                Step(tensors, board, values, i);

                result.Violations += Violations(values, i);
            }

            for (u16 h = 0; h < Graph.HeadCount; h++) Head(values, h, result);

            return result;
        }

    };

}

#endif //MANTARAY_REFERENCE_INTERPRETER_H
