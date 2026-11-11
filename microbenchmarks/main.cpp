//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#include <array>
#include <cstddef>
#include <cstdlib>
#include <type_traits>
#include <utility>
#include <vector>

#include <benchmark/benchmark.h>

#include <MantaRay/MantaRay.h>
#include <MantaRay/Reference/Position.h>

namespace BM = benchmark;

using namespace MantaRay;

template<s00 HiddenSize>
constexpr auto PerspectiveArchitecture()
{
    const auto [stm, ntm] = Input<Chess768>();

    const auto l0 = Affine(768           , HiddenSize, Weights<i16>(255, Clip(1.98)), Biases<i16>(255, Clip(1.98)));
    const auto l1 = Affine(HiddenSize * 2, 1         , Weights<i16>(64 , Clip(1.98)), Biases<i16>(255 * 64       ));

    const auto y0 = ClippedReLU(l0(stm));
    const auto y1 = ClippedReLU(l0(ntm));

    return Output(l1(Concat(y0, y1)), 400);
}

template<s00 HiddenSize>
constexpr auto SquaredPerspectiveArchitecture()
{
    const auto [stm, ntm] = Input<Chess768>();

    const auto l0 = Affine(768           , HiddenSize, Weights<i16>(255, Clip(1.98)), Biases<i16>(255, Clip(1.98)));
    const auto l1 = Affine(HiddenSize * 2, 1         , Weights<i16>(64 , Clip(1.98)), Biases<i16>(255 * 64       ),
                           DivideBy(255));

    const auto y0 = SquaredClippedReLU(l0(stm));
    const auto y1 = SquaredClippedReLU(l0(ntm));

    return Output(l1(Concat(y0, y1)), 400);
}

using Starshard = Network<PerspectiveArchitecture<256>>;
using    Aurora = Network<PerspectiveArchitecture<384>>;
using     Flare = Network<PerspectiveArchitecture<768>>;
using    Corona = Network<SquaredPerspectiveArchitecture< 768>>;
using   Eclipse = Network<SquaredPerspectiveArchitecture<1536>>;

template<s00 HiddenSize, UpdatePolicy Policy>
using Stack = AccumulatorStack<PerspectiveArchitecture<HiddenSize>, 512, Policy>;

template<s00 HiddenSize, UpdatePolicy Policy>
using SquaredStack = AccumulatorStack<SquaredPerspectiveArchitecture<HiddenSize>, 512, Policy>;

using StarshardStack = Stack<256, UpdatePolicy::Eager>;
using    AuroraStack = Stack<384, UpdatePolicy::Eager>;
using     FlareStack = Stack<768, UpdatePolicy::Eager>;
using    CoronaStack = SquaredStack< 768, UpdatePolicy::Eager>;
using   EclipseStack = SquaredStack<1536, UpdatePolicy::Eager>;

using StarshardLazyStack = Stack<256, UpdatePolicy::Lazy>;
using    AuroraLazyStack = Stack<384, UpdatePolicy::Lazy>;
using     FlareLazyStack = Stack<768, UpdatePolicy::Lazy>;
using    CoronaLazyStack = SquaredStack< 768, UpdatePolicy::Lazy>;
using   EclipseLazyStack = SquaredStack<1536, UpdatePolicy::Lazy>;

static_assert(Starshard::Fusable && Aurora::Fusable && Flare::Fusable && Corona::Fusable && Eclipse::Fusable);

using enum Piece;

constexpr u08 Square(const char (&name)[3]) { return static_cast<u08>(name[0] - 'a' + 8 * (name[1] - '1')); }

Reference::Position Root { "r1bqk2r/ppp2ppp/2np1n2/2b1p3/2B1P3/2PP1N2/PP3PPP/RNBQK2R w KQkq - 0 4" };

constexpr PieceDelta Insert = [] {
    PieceDelta delta;

    delta.Insert(Pawn, White, Square("a2"));

    return delta;
}();

constexpr PieceDelta Remove = [] {
    PieceDelta delta;

    delta.Remove(Pawn, White, Square("a2"));

    return delta;
}();

constexpr PieceDelta Move = [] {
    PieceDelta delta;

    delta.Move(Pawn, White, Square("a2"), Square("a4"));

    return delta;
}();

constexpr std::array<PieceDelta, 4> Line = [] {
    std::array<PieceDelta, 4> line;

    line[0].Move(Knight, White, Square("b1"), Square("d2"));

    line[1].Remove(Pawn, White, Square("e4"));
    line[1].Move(Knight, Black, Square("f6"), Square("e4"));

    line[2].Move(Pawn, White, Square("h2"), Square("h3"));

    line[3].Remove(Knight, White, Square("d2"));
    line[3].Move(Knight, Black, Square("e4"), Square("d2"));

    return line;
}();

class Random
{

    u64 State;

    public:
    explicit constexpr Random(const u64 seed) noexcept : State(seed) {}

    constexpr u64 Next() noexcept
    {
        u64 value = State += 0x9E3779B97F4A7C15;

        value = (value ^ value >> 30) * 0xBF58476D1CE4E5B9;
        value = (value ^ value >> 27) * 0x94D049BB133111EB;

        return value ^ value >> 31;
    }

};

template<auto D, u16 I>
void Append(std::vector<std::byte>& bytes, Random& random)
{
    using M = Model<D>;
    using V = IO::Tensors<M>::template Element<I>;
    using U = std::make_unsigned_t<V>;

    constexpr IR::Tensor Spec   = M::Graph.Tensors[I];
    constexpr Interval   Bounds = IR::BoundsOf(Spec);

    const u64 span = static_cast<u64>(Bounds.Max - Bounds.Min) + 1;

    for (s00 i = 0; i < IR::ElementsOf(Spec); i++) {
        const U bits = static_cast<U>(Bounds.Min + static_cast<i64>(random.Next() % span));

        for (s00 j = 0; j < sizeof(U); j++) bytes.push_back(static_cast<std::byte>(bits >> 8 * j));
    }
}

template<auto D>
void Load(Network<D>& network)
{
    Random random(0x4D616E7461526179);

    std::vector<std::byte> bytes;

    [&]<u16... I>(const std::integer_sequence<u16, I...>) {
        (Append<D, I>(bytes, random), ...);
    }(std::make_integer_sequence<u16, Model<D>::Graph.TensorCount> {});

    if (!network.Load(bytes)) std::abort();
}

Starshard StarshardNN;
Aurora       AuroraNN;
Flare         FlareNN;
Corona       CoronaNN;
Eclipse     EclipseNN;

StarshardStack StarshardAccumulatorStack { StarshardNN };
   AuroraStack    AuroraAccumulatorStack {    AuroraNN };
    FlareStack     FlareAccumulatorStack {     FlareNN };
   CoronaStack    CoronaAccumulatorStack {    CoronaNN };
  EclipseStack   EclipseAccumulatorStack {   EclipseNN };

StarshardLazyStack StarshardLazyAccumulatorStack { StarshardNN };
   AuroraLazyStack    AuroraLazyAccumulatorStack {    AuroraNN };
    FlareLazyStack     FlareLazyAccumulatorStack {     FlareNN };
   CoronaLazyStack    CoronaLazyAccumulatorStack {    CoronaNN };
  EclipseLazyStack   EclipseLazyAccumulatorStack {   EclipseNN };

[[maybe_unused]] const bool Ready = [] {
    Load(StarshardNN);
    Load(   AuroraNN);
    Load(    FlareNN);
    Load(   CoronaNN);
    Load(  EclipseNN);

    StarshardAccumulatorStack.Reset(Root);
       AuroraAccumulatorStack.Reset(Root);
        FlareAccumulatorStack.Reset(Root);
       CoronaAccumulatorStack.Reset(Root);
      EclipseAccumulatorStack.Reset(Root);

    StarshardLazyAccumulatorStack.Reset(Root);
       AuroraLazyAccumulatorStack.Reset(Root);
        FlareLazyAccumulatorStack.Reset(Root);
       CoronaLazyAccumulatorStack.Reset(Root);
      EclipseLazyAccumulatorStack.Reset(Root);

    return true;
}();

template<auto& S>
void BM_AccumulatorStack(BM::State& state)
{
    for (auto _ : state) {
        S.Push(PieceDelta {});

        BM::DoNotOptimize(S.Depth());
        BM::ClobberMemory();

        S.Pop();
    }
}

template<auto& S>
void BM_Refresh(BM::State& state)
{
    for (auto _ : state) {
        S.Reset(Root);

        BM::ClobberMemory();
    }
}

template<auto& S, const PieceDelta& D>
void BM_Update(BM::State& state)
{
    for (auto _ : state) {
        S.Push(D);

        BM::ClobberMemory();

        S.Pop();
    }
}

template<auto& S>
void BM_Evaluate(BM::State& state)
{
    for (auto _ : state) BM::DoNotOptimize(
        S.Evaluate(0)
    );
}

void Plies(BM::Benchmark* const benchmark) { benchmark->Arg(1)->Arg(2)->Arg(4); }

template<auto& S>
void BM_PushEvaluate(BM::State& state)
{
    const s00 plies = static_cast<s00>(state.range(0));
    const u08 stm   = static_cast<u08>(plies % 2);

    for (auto _ : state) {
        for (s00 ply = 0; ply < plies; ply++) S.Push(Line[ply]);

        BM::DoNotOptimize(S.Evaluate(stm));
        BM::ClobberMemory();

        for (s00 ply = 0; ply < plies; ply++) S.Pop();
    }
}

BENCHMARK(BM_AccumulatorStack<StarshardAccumulatorStack>)->Name("BM_AccumulatorStack_Starshard");
BENCHMARK(BM_AccumulatorStack<   AuroraAccumulatorStack>)->Name("BM_AccumulatorStack_Aurora");
BENCHMARK(BM_AccumulatorStack<    FlareAccumulatorStack>)->Name("BM_AccumulatorStack_Flare");
BENCHMARK(BM_AccumulatorStack<   CoronaAccumulatorStack>)->Name("BM_AccumulatorStack_Corona");
BENCHMARK(BM_AccumulatorStack<  EclipseAccumulatorStack>)->Name("BM_AccumulatorStack_Eclipse");

BENCHMARK(BM_Refresh<StarshardAccumulatorStack>)->Name("BM_Refresh_Starshard");
BENCHMARK(BM_Refresh<   AuroraAccumulatorStack>)->Name("BM_Refresh_Aurora");
BENCHMARK(BM_Refresh<    FlareAccumulatorStack>)->Name("BM_Refresh_Flare");
BENCHMARK(BM_Refresh<   CoronaAccumulatorStack>)->Name("BM_Refresh_Corona");
BENCHMARK(BM_Refresh<  EclipseAccumulatorStack>)->Name("BM_Refresh_Eclipse");

BENCHMARK(BM_Update<StarshardAccumulatorStack, Insert>)->Name("BM_Insert_Starshard");
BENCHMARK(BM_Update<   AuroraAccumulatorStack, Insert>)->Name("BM_Insert_Aurora");
BENCHMARK(BM_Update<    FlareAccumulatorStack, Insert>)->Name("BM_Insert_Flare");
BENCHMARK(BM_Update<   CoronaAccumulatorStack, Insert>)->Name("BM_Insert_Corona");
BENCHMARK(BM_Update<  EclipseAccumulatorStack, Insert>)->Name("BM_Insert_Eclipse");

BENCHMARK(BM_Update<StarshardAccumulatorStack, Remove>)->Name("BM_Remove_Starshard");
BENCHMARK(BM_Update<   AuroraAccumulatorStack, Remove>)->Name("BM_Remove_Aurora");
BENCHMARK(BM_Update<    FlareAccumulatorStack, Remove>)->Name("BM_Remove_Flare");
BENCHMARK(BM_Update<   CoronaAccumulatorStack, Remove>)->Name("BM_Remove_Corona");
BENCHMARK(BM_Update<  EclipseAccumulatorStack, Remove>)->Name("BM_Remove_Eclipse");

BENCHMARK(BM_Update<StarshardAccumulatorStack, Move>)->Name("BM_Move_Starshard");
BENCHMARK(BM_Update<   AuroraAccumulatorStack, Move>)->Name("BM_Move_Aurora");
BENCHMARK(BM_Update<    FlareAccumulatorStack, Move>)->Name("BM_Move_Flare");
BENCHMARK(BM_Update<   CoronaAccumulatorStack, Move>)->Name("BM_Move_Corona");
BENCHMARK(BM_Update<  EclipseAccumulatorStack, Move>)->Name("BM_Move_Eclipse");

BENCHMARK(BM_Evaluate<StarshardAccumulatorStack>)->Name("BM_Evaluate_Starshard");
BENCHMARK(BM_Evaluate<   AuroraAccumulatorStack>)->Name("BM_Evaluate_Aurora");
BENCHMARK(BM_Evaluate<    FlareAccumulatorStack>)->Name("BM_Evaluate_Flare");
BENCHMARK(BM_Evaluate<   CoronaAccumulatorStack>)->Name("BM_Evaluate_Corona");
BENCHMARK(BM_Evaluate<  EclipseAccumulatorStack>)->Name("BM_Evaluate_Eclipse");

BENCHMARK(BM_AccumulatorStack<StarshardLazyAccumulatorStack>)->Name("BM_AccumulatorStack_Lazy_Starshard");
BENCHMARK(BM_AccumulatorStack<   AuroraLazyAccumulatorStack>)->Name("BM_AccumulatorStack_Lazy_Aurora");
BENCHMARK(BM_AccumulatorStack<    FlareLazyAccumulatorStack>)->Name("BM_AccumulatorStack_Lazy_Flare");
BENCHMARK(BM_AccumulatorStack<   CoronaLazyAccumulatorStack>)->Name("BM_AccumulatorStack_Lazy_Corona");
BENCHMARK(BM_AccumulatorStack<  EclipseLazyAccumulatorStack>)->Name("BM_AccumulatorStack_Lazy_Eclipse");

BENCHMARK(BM_Evaluate<StarshardLazyAccumulatorStack>)->Name("BM_Evaluate_Lazy_Starshard");
BENCHMARK(BM_Evaluate<   AuroraLazyAccumulatorStack>)->Name("BM_Evaluate_Lazy_Aurora");
BENCHMARK(BM_Evaluate<    FlareLazyAccumulatorStack>)->Name("BM_Evaluate_Lazy_Flare");
BENCHMARK(BM_Evaluate<   CoronaLazyAccumulatorStack>)->Name("BM_Evaluate_Lazy_Corona");
BENCHMARK(BM_Evaluate<  EclipseLazyAccumulatorStack>)->Name("BM_Evaluate_Lazy_Eclipse");

BENCHMARK(BM_PushEvaluate<StarshardAccumulatorStack>)->Name("BM_PushEvaluate_Eager_Starshard")->Apply(Plies);
BENCHMARK(BM_PushEvaluate<   AuroraAccumulatorStack>)->Name("BM_PushEvaluate_Eager_Aurora"   )->Apply(Plies);
BENCHMARK(BM_PushEvaluate<    FlareAccumulatorStack>)->Name("BM_PushEvaluate_Eager_Flare"    )->Apply(Plies);
BENCHMARK(BM_PushEvaluate<   CoronaAccumulatorStack>)->Name("BM_PushEvaluate_Eager_Corona"   )->Apply(Plies);
BENCHMARK(BM_PushEvaluate<  EclipseAccumulatorStack>)->Name("BM_PushEvaluate_Eager_Eclipse"  )->Apply(Plies);

BENCHMARK(BM_PushEvaluate<StarshardLazyAccumulatorStack>)->Name("BM_PushEvaluate_Lazy_Starshard")->Apply(Plies);
BENCHMARK(BM_PushEvaluate<   AuroraLazyAccumulatorStack>)->Name("BM_PushEvaluate_Lazy_Aurora"   )->Apply(Plies);
BENCHMARK(BM_PushEvaluate<    FlareLazyAccumulatorStack>)->Name("BM_PushEvaluate_Lazy_Flare"    )->Apply(Plies);
BENCHMARK(BM_PushEvaluate<   CoronaLazyAccumulatorStack>)->Name("BM_PushEvaluate_Lazy_Corona"   )->Apply(Plies);
BENCHMARK(BM_PushEvaluate<  EclipseLazyAccumulatorStack>)->Name("BM_PushEvaluate_Lazy_Eclipse"  )->Apply(Plies);

BENCHMARK_MAIN();
