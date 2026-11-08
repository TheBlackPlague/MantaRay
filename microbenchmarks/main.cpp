//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#include <array>
#include <cstddef>
#include <cstdlib>
#include <string_view>
#include <type_traits>
#include <vector>

#include <benchmark/benchmark.h>

#include <MantaRay/MantaRay.h>

namespace BM = benchmark;

using namespace MantaRay;

template<s00 HiddenSize>
using PerspectiveArchitecture = Architecture<
    FeatureTransformer<Chess768, HiddenSize,
        Weights<i16, 255, Clip<-1.98, 1.98>>,
        Biases <i16, 255, Clip<-1.98, 1.98>>
    >,
    ClippedReLU<0, 1>,
    Concat,
    Affine<HiddenSize * 2, 1,
        Weights<i16,  64     , Clip<-1.98, 1.98>>,
        Biases <i16, 255 * 64>
    >,
    Output<400>
>;

using Starshard = Network<PerspectiveArchitecture<256>>;
using    Aurora = Network<PerspectiveArchitecture<384>>;
using     Flare = Network<PerspectiveArchitecture<768>>;

template<s00 HiddenSize, UpdatePolicy Policy>
using Stack = AccumulatorStack<PerspectiveArchitecture<HiddenSize>, 512, Policy>;

using StarshardStack = Stack<256, UpdatePolicy::Eager>;
using    AuroraStack = Stack<384, UpdatePolicy::Eager>;
using     FlareStack = Stack<768, UpdatePolicy::Eager>;

using StarshardLazyStack = Stack<256, UpdatePolicy::Lazy>;
using    AuroraLazyStack = Stack<384, UpdatePolicy::Lazy>;
using     FlareLazyStack = Stack<768, UpdatePolicy::Lazy>;

static_assert(StarshardLazyStack::Fused && AuroraLazyStack::Fused && FlareLazyStack::Fused);

enum Piece : u08 { Pawn, Knight, Bishop, Rook, Queen, King };
enum Color : u08 { White, Black };

using Boards = std::array<std::array<u64, 2>, 6>;

constexpr u08 Square(const char (&name)[3]) { return static_cast<u08>(name[0] - 'a' + 8 * (name[1] - '1')); }

constexpr Boards ParsePlacement(const std::string_view placement)
{
    constexpr std::string_view Letters = "pnbrqk";

    Boards boards {};

    s00 file = 0;
    s00 rank = 7;

    for (const char symbol : placement) {
        if (symbol == '/') {
            file = 0;
            rank--;
        } else if (symbol >= '1' && symbol <= '8') file += static_cast<s00>(symbol - '0');
        else {
            const u08 color = symbol >= 'a' ? Black : White;
            const u08 piece = static_cast<u08>(Letters.find(static_cast<char>(symbol | 0x20)));

            boards[piece][color] |= u64 { 1 } << (rank * 8 + file++);
        }
    }

    return boards;
}

Boards Root = ParsePlacement("r1bqk2r/ppp2ppp/2np1n2/2b1p3/2B1P3/2PP1N2/PP3PPP/RNBQK2R");

constexpr auto RootBoards = [](const u08 piece, const u08 color) noexcept { return Root[piece][color]; };

constexpr Delta Insert = [] {
    Delta delta;

    delta.Insert(Pawn, White, Square("a2"));

    return delta;
}();

constexpr Delta Remove = [] {
    Delta delta;

    delta.Remove(Pawn, White, Square("a2"));

    return delta;
}();

constexpr Delta Move = [] {
    Delta delta;

    delta.Move(Pawn, White, Square("a2"), Square("a4"));

    return delta;
}();

constexpr std::array<Delta, 4> Line = [] {
    std::array<Delta, 4> line;

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

template<typename S>
void Append(std::vector<std::byte>& bytes, Random& random, const s00 count)
{
    using U = std::make_unsigned_t<typename S::Type>;

    const u64 span = static_cast<u64>(S::Bounds.Max - S::Bounds.Min) + 1;

    for (s00 i = 0; i < count; i++) {
        const U bits = static_cast<U>(S::Bounds.Min + static_cast<i64>(random.Next() % span));

        for (s00 j = 0; j < sizeof(U); j++) bytes.push_back(static_cast<std::byte>(bits >> 8 * j));
    }
}

template<typename A>
void Load(Network<A>& network)
{
    using FT = A::Transformer;
    using D  = Backend::Stage::LayerAt<A, 2>;

    Random random(0x4D616E7461526179);

    std::vector<std::byte> bytes;

    Append<typename FT::WeightSpec>(bytes, random, FT::Features::InputSize * FT::OutputSize);
    Append<typename FT::BiasSpec  >(bytes, random,                           FT::OutputSize);
    Append<typename D ::WeightSpec>(bytes, random, D::InputSize * D::OutputSize);
    Append<typename D ::BiasSpec  >(bytes, random,                D::OutputSize);

    if (!network.Load(bytes)) std::abort();
}

Starshard StarshardNN;
Aurora       AuroraNN;
Flare         FlareNN;

StarshardStack StarshardAccumulatorStack { StarshardNN };
   AuroraStack    AuroraAccumulatorStack {    AuroraNN };
    FlareStack     FlareAccumulatorStack {     FlareNN };

StarshardLazyStack StarshardLazyAccumulatorStack { StarshardNN };
   AuroraLazyStack    AuroraLazyAccumulatorStack {    AuroraNN };
    FlareLazyStack     FlareLazyAccumulatorStack {     FlareNN };

[[maybe_unused]] const bool Ready = [] {
    Load(StarshardNN);
    Load(   AuroraNN);
    Load(    FlareNN);

    StarshardAccumulatorStack.Reset(RootBoards);
       AuroraAccumulatorStack.Reset(RootBoards);
        FlareAccumulatorStack.Reset(RootBoards);

    StarshardLazyAccumulatorStack.Reset(RootBoards);
       AuroraLazyAccumulatorStack.Reset(RootBoards);
        FlareLazyAccumulatorStack.Reset(RootBoards);

    return true;
}();

template<auto& S>
void BM_AccumulatorStack(BM::State& state)
{
    for (auto _ : state) {
        S.Push(Delta {});

        BM::DoNotOptimize(S.Depth());
        BM::ClobberMemory();

        S.Pop();
    }
}

template<auto& S>
void BM_Refresh(BM::State& state)
{
    for (auto _ : state) {
        S.Reset(RootBoards);

        BM::ClobberMemory();
    }
}

template<auto& S, const Delta& D>
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

void Plies(BM::Benchmark* benchmark) { benchmark->Arg(1)->Arg(2)->Arg(4); }

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

BENCHMARK(BM_Refresh<StarshardAccumulatorStack>)->Name("BM_Refresh_Starshard");
BENCHMARK(BM_Refresh<   AuroraAccumulatorStack>)->Name("BM_Refresh_Aurora");
BENCHMARK(BM_Refresh<    FlareAccumulatorStack>)->Name("BM_Refresh_Flare");

BENCHMARK(BM_Update<StarshardAccumulatorStack, Insert>)->Name("BM_Insert_Starshard");
BENCHMARK(BM_Update<   AuroraAccumulatorStack, Insert>)->Name("BM_Insert_Aurora");
BENCHMARK(BM_Update<    FlareAccumulatorStack, Insert>)->Name("BM_Insert_Flare");

BENCHMARK(BM_Update<StarshardAccumulatorStack, Remove>)->Name("BM_Remove_Starshard");
BENCHMARK(BM_Update<   AuroraAccumulatorStack, Remove>)->Name("BM_Remove_Aurora");
BENCHMARK(BM_Update<    FlareAccumulatorStack, Remove>)->Name("BM_Remove_Flare");

BENCHMARK(BM_Update<StarshardAccumulatorStack, Move>)->Name("BM_Move_Starshard");
BENCHMARK(BM_Update<   AuroraAccumulatorStack, Move>)->Name("BM_Move_Aurora");
BENCHMARK(BM_Update<    FlareAccumulatorStack, Move>)->Name("BM_Move_Flare");

BENCHMARK(BM_Evaluate<StarshardAccumulatorStack>)->Name("BM_Evaluate_Starshard");
BENCHMARK(BM_Evaluate<   AuroraAccumulatorStack>)->Name("BM_Evaluate_Aurora");
BENCHMARK(BM_Evaluate<    FlareAccumulatorStack>)->Name("BM_Evaluate_Flare");

BENCHMARK(BM_AccumulatorStack<StarshardLazyAccumulatorStack>)->Name("BM_AccumulatorStack_Lazy_Starshard");
BENCHMARK(BM_AccumulatorStack<   AuroraLazyAccumulatorStack>)->Name("BM_AccumulatorStack_Lazy_Aurora");
BENCHMARK(BM_AccumulatorStack<    FlareLazyAccumulatorStack>)->Name("BM_AccumulatorStack_Lazy_Flare");

BENCHMARK(BM_Evaluate<StarshardLazyAccumulatorStack>)->Name("BM_Evaluate_Lazy_Starshard");
BENCHMARK(BM_Evaluate<   AuroraLazyAccumulatorStack>)->Name("BM_Evaluate_Lazy_Aurora");
BENCHMARK(BM_Evaluate<    FlareLazyAccumulatorStack>)->Name("BM_Evaluate_Lazy_Flare");

BENCHMARK(BM_PushEvaluate<StarshardAccumulatorStack>)->Name("BM_PushEvaluate_Eager_Starshard")->Apply(Plies);
BENCHMARK(BM_PushEvaluate<   AuroraAccumulatorStack>)->Name("BM_PushEvaluate_Eager_Aurora"   )->Apply(Plies);
BENCHMARK(BM_PushEvaluate<    FlareAccumulatorStack>)->Name("BM_PushEvaluate_Eager_Flare"    )->Apply(Plies);

BENCHMARK(BM_PushEvaluate<StarshardLazyAccumulatorStack>)->Name("BM_PushEvaluate_Lazy_Starshard")->Apply(Plies);
BENCHMARK(BM_PushEvaluate<   AuroraLazyAccumulatorStack>)->Name("BM_PushEvaluate_Lazy_Aurora"   )->Apply(Plies);
BENCHMARK(BM_PushEvaluate<    FlareLazyAccumulatorStack>)->Name("BM_PushEvaluate_Lazy_Flare"    )->Apply(Plies);

BENCHMARK_MAIN();
