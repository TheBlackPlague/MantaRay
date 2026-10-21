//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_RUNTIME_NETWORKUTIL_H
#define MANTARAY_RUNTIME_NETWORKUTIL_H

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <memory>
#include <random>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <MantaRay/Architecture/FeatureSet.h>
#include <MantaRay/Backend/Stage/Format.h>
#include <MantaRay/Backend/Target.h>
#include <MantaRay/Common/List.h>
#include <MantaRay/IO/Loader.h>
#include <MantaRay/Runtime/AccumulatorStack.h>
#include <MantaRay/Runtime/Delta.h>
#include <MantaRay/Runtime/Network.h>

using namespace MantaRay;

enum class Fill : u08 { Random, Maximum, Minimum, Extremes };

template<typename A>
constexpr bool Bucketed = A::Transformer::Features::KeyCount > 1;

template<typename A>
constexpr bool NeedsKings = Bucketed<A> || Network<A>::Buckets::Count > 1;

template<typename A, UpdatePolicy Policy, s00 Capacity = 33, typename ISA = Backend::Native>
using Stack = AccumulatorStack<A, Capacity, Policy, ISA>;

inline i64 Uniform(std::mt19937_64& random, const i64 min, const i64 max)
{
    return min + static_cast<i64>(random() % static_cast<u64>(max - min + 1));
}

template<typename S>
void Append(std::vector<std::byte>& bytes, std::mt19937_64& random, const s00 count, const Fill fill)
{
    using V = S::Type;
    using U = std::conditional_t<sizeof(V) == 1, u08, std::conditional_t<sizeof(V) == 2, u16, u32>>;

    for (s00 i = 0; i < count; i++) {
        const bool high = fill == Fill::Maximum || (fill == Fill::Extremes && random() % 2 == 0);

        V value;

        if constexpr (std::is_same_v<V, f32>) {
            const f32 real = static_cast<f32>(random() >> 40) / 8388608.0F - 1.0F;

            value = fill == Fill::Random ? real : high ? 1.0F : -1.0F;
        } else {
            const i64 uniform = Uniform(random, S::Bounds.Min, S::Bounds.Max);

            value = static_cast<V>(fill == Fill::Random ? uniform : high ? S::Bounds.Max : S::Bounds.Min);
        }

        const U bits = std::bit_cast<U>(value);

        for (s00 j = 0; j < sizeof(U); j++) bytes.push_back(static_cast<std::byte>(bits >> 8 * j));
    }
}

template<typename A>
std::vector<std::byte> RandomBytes(
          std::mt19937_64& random                    ,
    const Fill             transformer = Fill::Random,
    const Fill             dense       = Fill::Random
)
{
    using FT = A::Transformer;

    std::vector<std::byte> bytes;

    Append<typename FT::WeightSpec>(bytes, random, FT::Features::InputSize * FT::OutputSize, transformer);
    Append<typename FT::BiasSpec  >(bytes, random,                           FT::OutputSize, transformer);

    [&]<typename... L>(List<L...>) {
        ([&] {
            if constexpr (Backend::Stage::IsAffine<L>) {
                Append<typename L::WeightSpec>(bytes, random, L::InputSize * L::OutputSize, dense);
                Append<typename L::BiasSpec  >(bytes, random,                L::OutputSize, dense);
            }
        }(), ...);
    }(typename A::Layers {});

    return bytes;
}

inline void Pad(std::vector<std::byte>& bytes)
{
    const s00 payload = bytes.size();

    for (s00 i = payload; i < IO::PaddedSize(payload); i++)
        bytes.push_back(static_cast<std::byte>(IO::Padding[(i - payload) % IO::Padding.size()]));
}

template<typename A, typename ISA = Backend::Native>
std::unique_ptr<Network<A, ISA>> RandomNetwork(std::mt19937_64& random)
{
    auto network = std::make_unique<Network<A, ISA>>();

    EXPECT_TRUE(static_cast<bool>(network->Load(RandomBytes<A>(random))));

    return network;
}

inline std::vector<u16> StartPosition()
{
    constexpr std::array<u08, 8> Back { 3, 1, 2, 4, 5, 2, 1, 3 };

    std::vector<u16> pieces;

    for (u08 file = 0; file < 8; file++) {
        pieces.push_back(PieceCode(Back[file], 0,      file));
        pieces.push_back(PieceCode(0         , 0,  8 + file));
        pieces.push_back(PieceCode(0         , 1, 48 + file));
        pieces.push_back(PieceCode(Back[file], 1, 56 + file));
    }

    return pieces;
}

inline std::array<u64, 12> PieceBoards(const std::span<const u16> pieces)
{
    std::array<u64, 12> boards {};

    for (const u16 code : pieces) boards[ColorOf(code) * 6 + PieceOf(code)] |= u64 { 1 } << SquareOf(code);

    return boards;
}

template<typename S>
void ResetFromPieces(S& stack, const std::span<const u16> pieces)
{
    const std::array<u64, 12> boards = PieceBoards(pieces);

    stack.Reset([&](const u08 piece, const u08 color) { return boards[color * 6 + piece]; });
}

template<typename S>
i32 EvaluateWithBoards(S& stack, const std::span<const u16> pieces, const u08 stm)
{
    const std::array<u64, 12> boards = PieceBoards(pieces);

    return stack.Evaluate(stm, [&](const u08 piece, const u08 color) { return boards[color * 6 + piece]; });
}

inline std::vector<u16> Apply(std::vector<u16> pieces, const Delta& delta)
{
    for (s00 i = 0; i < delta.RemovedCount(); i++) {
        const auto found = std::ranges::find(pieces, delta.Removed(i));

        EXPECT_NE(found, pieces.end()) << "removed piece " << delta.Removed(i) << " is absent";

        if (found != pieces.end()) pieces.erase(found);
    }

    for (s00 i = 0; i < delta.InsertedCount(); i++) pieces.push_back(delta.Inserted(i));

    return pieces;
}

template<typename S>
testing::AssertionResult SameValues(
    const S&                                       stack      ,
    const std::span<const typename S::Accumulator> expected   ,
    const u08                                      perspective
)
{
    const typename S::Accumulator* values = stack.Values(perspective);

    for (s00 i = 0; i < expected.size(); i++)
        if (values[i] != expected[i])
            return testing::AssertionFailure() << "perspective " << s00 { perspective } << ", index " << i << ": "
                                               << values[i] << " != " << expected[i];

    return testing::AssertionSuccess();
}

template<typename A>
class Oracle
{

    constexpr static s00 H = Network<A>::HiddenSize;

    const Network<A>& Model;

    std::unique_ptr<Stack<A, UpdatePolicy::Lazy, 1>> Fresh;

    s00 Calls = 0;

    public:
    explicit Oracle(const Network<A>& network) :
        Model(network), Fresh(std::make_unique<Stack<A, UpdatePolicy::Lazy, 1>>(network)) {}

    template<typename S>
    testing::AssertionResult Check(S& stack, const std::span<const u16> pieces, const u08 stm)
    {
        i32 actual = 0;

        if constexpr (Bucketed<A>) actual = EvaluateWithBoards(stack, pieces, stm);
        else actual = Calls++ % 2 == 0 ? stack.Evaluate(stm) : EvaluateWithBoards(stack, pieces, stm);

        const i32 expected = Model.Evaluate(pieces, stm);

        if (actual != expected)
            return testing::AssertionFailure() << "stm " << s00 { stm } << ": " << actual << " != " << expected;

        Fresh->ClearCache();

        ResetFromPieces(*Fresh, pieces);

        for (u08 perspective = 0; perspective < 2; perspective++) {
            const std::span<const typename S::Accumulator> values(Fresh->Values(perspective), H);

            if (testing::AssertionResult result = SameValues(stack, values, perspective); !result) return result;
        }

        return testing::AssertionSuccess();
    }

};

class Walk
{

    std::mt19937_64& Random;

    bool Kings;

    std::vector<std::vector<u16>> Positions;
    std::vector<bool            > Evaluated;

    u16 Pick(const std::vector<u16>& pieces) { return pieces[Random() % pieces.size()]; }

    u08 EmptySquare(const std::vector<u16>& pieces)
    {
        u64 occupied = 0;

        for (const u16 code : pieces) occupied |= u64 { 1 } << SquareOf(code);

        for (;;) {
            const auto square = static_cast<u08>(Random() % 64);

            if (!(occupied >> square & 1)) return square;
        }
    }

    static bool IsKing(const u16 code) { return PieceOf(code) == KingPiece; }

    static std::vector<u16> Others(const std::vector<u16>& pieces, const u08 color)
    {
        std::vector<u16> others;

        for (const u16 code : pieces)
            if (!IsKing(code) && (color > 1 || ColorOf(code) == color)) others.push_back(code);

        return others;
    }

    static u16 KingOf(const std::vector<u16>& pieces, const u08 color)
    {
        return *std::ranges::find_if(pieces, [&](const u16 code) { return IsKing(code) && ColorOf(code) == color; });
    }

    Delta Unconstrained(std::vector<u16>& pieces)
    {
        constexpr std::array<std::array<u08, 2>, 12> Shapes {{
            { 1, 1 }, { 1, 1 }, { 1, 1 }, { 2, 1 }, { 2, 1 }, { 2, 2 },
            { 0, 0 }, { 0, 1 }, { 0, 2 }, { 1, 0 }, { 1, 2 }, { 2, 0 }
        }};

        const std::array<u08, 2> shape = Shapes[Random() % Shapes.size()];

        const s00 removals  = std::min<s00>(shape[0], pieces.size());
        const s00 additions = std::min<s00>(shape[1], 32 - (pieces.size() - removals));

        Delta delta;

        std::array<u16, 2> removed {};

        for (s00 i = 0; i < removals; i++) {
            removed[i] = Pick(pieces);

            pieces.erase(std::ranges::find(pieces, removed[i]));

            delta.Remove(PieceOf(removed[i]), ColorOf(removed[i]), SquareOf(removed[i]));
        }

        for (s00 i = 0; i < additions; i++) {
            const bool moved  = i < removals && Random() % 4 != 0;
            const u08  piece  = moved ? PieceOf(removed[i]) : static_cast<u08>(Random() % 6);
            const u08  color  = moved ? ColorOf(removed[i]) : static_cast<u08>(Random() % 2);
            const u08  square = EmptySquare(pieces);

            pieces.push_back(PieceCode(piece, color, square));

            delta.Insert(piece, color, square);
        }

        return delta;
    }

    Delta WithKings(std::vector<u16>& pieces)
    {
        Delta delta;

        const auto remove = [&](const u16 code) {
            pieces.erase(std::ranges::find(pieces, code));

            delta.Remove(PieceOf(code), ColorOf(code), SquareOf(code));
        };

        const auto add = [&](const u08 piece, const u08 color, const u08 square) {
            pieces.push_back(PieceCode(piece, color, square));

            delta.Insert(piece, color, square);
        };

        const auto vacate = [&](const s00 count) {
            for (s00 i = 0; i < count && !Others(pieces, 2).empty(); i++) remove(Pick(Others(pieces, 2)));
        };

        const auto drop = [&] {
            add(static_cast<u08>(Random() % 5), static_cast<u08>(Random() % 2), EmptySquare(pieces));
        };

        const auto color = static_cast<u08>(Random() % 2);

        switch (Random() % 10) {
            case 0: case 1: case 2: {
                remove(KingOf(pieces, color));
                add   (KingPiece, color, EmptySquare(pieces));

                break;
            }
            case 3: case 4: {
                if (Others(pieces, 2).empty()) break;

                const u16 code = Pick(Others(pieces, 2));

                remove(code);
                add   (PieceOf(code), ColorOf(code), EmptySquare(pieces));

                break;
            }
            case 5: {
                if (Others(pieces, 2).empty()) break;

                const u16 victim = Pick(Others(pieces, 2));

                remove(victim);

                const u16 mover = Pick(pieces);

                remove(mover);
                add   (PieceOf(mover), ColorOf(mover), SquareOf(victim));

                break;
            }
            case 6: {
                if (Others(pieces, color).empty()) break;

                const u16 rook = Pick(Others(pieces, color));

                remove(KingOf(pieces, color));
                remove(rook);

                if (Random() % 2 == 0) {
                    add(KingPiece    , color, EmptySquare(pieces));
                    add(PieceOf(rook), color, EmptySquare(pieces));
                } else {
                    add(PieceOf(rook), color, EmptySquare(pieces));
                    add(KingPiece    , color, EmptySquare(pieces));
                }

                break;
            }
            case 7: {
                if (Others(pieces, 2).empty()) break;

                vacate(1 + Random() % 2);
                drop  ();

                break;
            }
            case 8: {
                for (s00 i = 1 + Random() % 2; i > 0 && pieces.size() < 32; i--) drop();

                break;
            }
            default: {
                vacate(Random() % 3);

                break;
            }
        }

        return delta;
    }

    public:
    explicit Walk(std::mt19937_64& random, const bool kings) : Random(random), Kings(kings) {}

    void Reset(std::vector<u16> pieces)
    {
        Positions = { std::move(pieces) };
        Evaluated = { true };
    }

    const std::vector<u16>& Pieces() const { return Positions.back(); }

    s00 Depth() const { return Positions.size() - 1; }

    Delta Next()
    {
        std::vector<u16> pieces = Pieces();

        const Delta delta = Kings ? WithKings(pieces) : Unconstrained(pieces);

        Positions.push_back(std::move(pieces));
        Evaluated.push_back(false);

        return delta;
    }

    Delta Null()
    {
        Positions.push_back(Pieces());
        Evaluated.push_back(false);

        return {};
    }

    void Pop()
    {
        Positions.pop_back();
        Evaluated.pop_back();
    }

    s00 Evaluate()
    {
        s00 pending = 0;

        while (!Evaluated[Evaluated.size() - 1 - pending]) pending++;

        std::ranges::fill(Evaluated, true);

        return pending;
    }

    std::vector<u16> RandomPosition()
    {
        std::vector<u16> pieces;

        const s00 count = 2 + Random() % 31;

        for (u08 color = 0; Kings && color < 2; color++)
            pieces.push_back(PieceCode(KingPiece, color, EmptySquare(pieces)));

        while (pieces.size() < count) {
            const auto piece = static_cast<u08>(Random() % (Kings ? 5 : 6));
            const auto color = static_cast<u08>(Random() % 2);

            pieces.push_back(PieceCode(piece, color, EmptySquare(pieces)));
        }

        return pieces;
    }

};

template<typename A>
std::array<u08, 2> KeysOf(const std::vector<u16>& pieces)
{
    using F = A::Transformer::Features;

    std::array<u08, 2> keys {};

    for (const u16 code : pieces)
        if (PieceOf(code) == KingPiece) keys[ColorOf(code)] = F::Key(ColorOf(code), SquareOf(code));

    return keys;
}

template<typename A>
void LazyEqualsEagerEqualsReset(const u64 seed, const s00 operations)
{
    constexpr s00 MaxDepth = 32;

    std::mt19937_64 random(seed);

    const auto network = RandomNetwork<A>(random);

    const auto lazy  = std::make_unique<Stack<A, UpdatePolicy::Lazy , MaxDepth + 1>>(*network);
    const auto eager = std::make_unique<Stack<A, UpdatePolicy::Eager, MaxDepth + 1>>(*network);

    Oracle<A> oracle(*network);

    Walk walk(random, NeedsKings<A>);

    walk.Reset(StartPosition());

    ResetFromPieces(*lazy , walk.Pieces());
    ResetFromPieces(*eager, walk.Pieces());

    std::array<s00, 3> pending {};

    s00 pushes = 0, pops = 0, evaluations = 0, resets = 0, crossings = 0;

    for (s00 operation = 0; operation < operations; operation++) {
        const u64 roll = random() % 100;

        if (roll < 45 && walk.Depth() < MaxDepth) {
            const std::array<u08, 2> keys = KeysOf<A>(walk.Pieces());

            const Delta delta = random() % 16 == 0 ? walk.Null() : walk.Next();

            lazy ->Push(delta);
            eager->Push(delta);

            pushes   ++;
            crossings += KeysOf<A>(walk.Pieces()) != keys;
        } else if (roll < 75 && walk.Depth() > 0) {
            walk.Pop();

            lazy ->Pop();
            eager->Pop();

            pops++;
        } else if (roll < 77) {
            walk.Reset(walk.RandomPosition());

            ResetFromPieces(*lazy , walk.Pieces());
            ResetFromPieces(*eager, walk.Pieces());

            resets++;
        } else {
            pending[std::min<s00>(walk.Evaluate(), 2)]++;

            const auto stm = static_cast<u08>(random() % 2);

            ASSERT_TRUE(oracle.Check(*lazy , walk.Pieces(), stm)) << "lazy, operation " << operation;
            ASSERT_TRUE(oracle.Check(*eager, walk.Pieces(), stm)) << "eager, operation " << operation;

            if (random() % 4 == 0) {
                ASSERT_TRUE(oracle.Check(*lazy , walk.Pieces(), stm ^ 1)) << "lazy, operation " << operation;
                ASSERT_TRUE(oracle.Check(*eager, walk.Pieces(), stm ^ 1)) << "eager, operation " << operation;
            }

            evaluations++;
        }

        ASSERT_EQ(lazy ->Depth(), walk.Depth());
        ASSERT_EQ(eager->Depth(), walk.Depth());
    }

    EXPECT_GT(pushes     , operations / 3);
    EXPECT_GT(pops       , operations / 4);
    EXPECT_GT(resets     , s00 { 0 });
    EXPECT_GT(evaluations, operations / 5);
    EXPECT_GT(pending[0] , s00 { 0 });
    EXPECT_GT(pending[1] , s00 { 0 });
    EXPECT_GT(pending[2] , s00 { 0 });

    if constexpr (Bucketed<A>) EXPECT_GT(crossings, pushes / 10);
}

#endif //MANTARAY_RUNTIME_NETWORKUTIL_H
