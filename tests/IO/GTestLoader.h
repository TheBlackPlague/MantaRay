//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_IO_GTESTLOADER_H
#define MANTARAY_IO_GTESTLOADER_H

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <memory>
#include <random>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <MantaRay/Backend/Lowering.h>
#include <MantaRay/Backend/Target.h>
#include <MantaRay/IO/Loader.h>
#include <MantaRay/Runtime/Network.h>

#include "../Architecture/Examples.h"
#include "../Runtime/NetworkUtil.h"

using namespace MantaRay;

namespace LoaderTest
{

    using A = Examples::Aurora;

    constexpr s00 H = A::Transformer::OutputSize;

    constexpr s00 L0 = 0;
    constexpr s00 B0 = L0 + 768 * H * sizeof(i16);
    constexpr s00 L1 = B0 +       H * sizeof(i16);
    constexpr s00 B1 = L1 + 2 * H * sizeof(i16);

    constexpr s00 Payload = B1 + sizeof(i16);
    constexpr s00 Padded  = IO::PaddedSize(Payload);

    static_assert(Payload == 592130 && Padded == 592192);

    template<typename T>
    std::vector<std::byte> Bytes(const T& object)
    {
        const std::span<const std::byte> bytes = std::as_bytes(std::span(&object, 1));

        return { bytes.begin(), bytes.end() };
    }

    template<typename T>
    void Patch(std::vector<std::byte>& bytes, const s00 offset, const T value)
    {
        using U = std::make_unsigned_t<T>;

        const U bits = std::bit_cast<U>(value);

        for (s00 i = 0; i < sizeof(T); i++) bytes[offset + i] = static_cast<std::byte>(bits >> 8 * i);
    }

    inline std::vector<std::vector<u16>> Positions(std::mt19937_64& random, const bool kings = false)
    {
        Walk walk(random, kings);

        std::vector<std::vector<u16>> positions { StartPosition() };

        for (s00 i = 0; i < 64; i++) positions.push_back(walk.RandomPosition());

        return positions;
    }

    template<typename N>
    std::vector<i32> Evaluations(const N& network, const std::vector<std::vector<u16>>& positions)
    {
        std::vector<i32> values;

        for (const std::vector<u16>& pieces : positions)
        for (u08 stm = 0; stm < 2; stm++) values.push_back(network.Evaluate(pieces, stm));

        return values;
    }

}

TEST(Loader, ReaderIsLittleEndian)
{
    const std::vector<std::byte> bytes {
        std::byte { 0x01 }, std::byte { 0x02 }, std::byte { 0xFF }, std::byte { 0x7F },
        std::byte { 0x00 }, std::byte { 0x00 }, std::byte { 0x80 }, std::byte { 0xBF },
        std::byte { 0x85 }
    };

    IO::Reader reader(bytes);

    EXPECT_EQ(reader.Remaining(), s00 { 9 });
    EXPECT_EQ(reader.Tensor   (), 0);

    std::array<i16, 2> pair  {};
    std::array<f32, 1> real  {};
    std::array<i08, 1> small {};
    std::array<i16, 1> extra {};

    ASSERT_TRUE(reader.Read(std::span<i16>(pair)));
    EXPECT_EQ(pair[0], 0x0201);
    EXPECT_EQ(pair[1], 0x7FFF);

    ASSERT_TRUE(reader.Read(std::span<f32>(real)));
    EXPECT_EQ(real[0], -1.0F);
    EXPECT_EQ(reader.Remaining(), s00 { 1 });
    EXPECT_EQ(reader.Tensor   (), 2);

    EXPECT_FALSE(reader.Read(std::span<i16>(extra)));
    EXPECT_EQ(reader.Remaining(), s00 { 1 });
    EXPECT_EQ(reader.Tensor   (), 2);

    ASSERT_TRUE(reader.Read(std::span<i08>(small)));
    EXPECT_EQ(small[0], -123);
    EXPECT_EQ(reader.Remaining(), s00 { 0 });
    EXPECT_EQ(reader.Tensor   (), 3);
}

TEST(Loader, StatusMessages)
{
    static_assert(LoadResult {                            }.Message() == "ok"             );
    static_assert(LoadResult { LoadStatus::SizeMismatch   }.Message() == "size mismatch"  );
    static_assert(LoadResult { LoadStatus::BadPadding     }.Message() == "bad padding"    );
    static_assert(LoadResult { LoadStatus::BoundViolation }.Message() == "bound violation");

    static_assert( static_cast<bool>(LoadResult {                          }));
    static_assert(!static_cast<bool>(LoadResult { LoadStatus::SizeMismatch }));

    std::vector<std::string_view> messages;

    for (u08 status = 0; status <= static_cast<u08>(LoadStatus::BoundViolation); status++)
        messages.push_back(LoadResult { static_cast<LoadStatus>(status) }.Message());

    std::ranges::sort(messages);

    EXPECT_EQ(std::ranges::adjacent_find(messages), messages.end());
    EXPECT_EQ(LoadResult { static_cast<LoadStatus>(0xFF) }.Message(), "unknown status");

    const auto network = std::make_unique<Network<LoaderTest::A>>();

    EXPECT_EQ(network->Load({}).Message(), "size mismatch");
}

TEST(Loader, PaddingVariants)
{
    using namespace LoaderTest;

    std::mt19937_64 random(0x50414444494E4753);

    const std::vector<std::vector<u16>> positions = Positions(random);

    const std::vector<std::byte> payload = RandomBytes<A>(random);
          std::vector<std::byte> bytes   = payload;

    Pad(bytes);

    ASSERT_EQ(payload.size(), Payload);
    ASSERT_EQ(bytes  .size(), Padded );

    const auto network = std::make_unique<Network<A>>();

    ASSERT_EQ(network->Load(payload).Status, LoadStatus::Ok);

    const std::vector<i32> expected = Evaluations(*network, positions);

    ASSERT_EQ(network->Load(bytes).Status, LoadStatus::Ok);
    EXPECT_EQ(Evaluations(*network, positions), expected);

    for (const s00 offset : { Payload, Payload + 31, Padded - 1 }) {
        std::vector<std::byte> patched = bytes;

        patched[offset] ^= std::byte { 0x20 };

        const LoadResult result = network->Load(patched);

        EXPECT_EQ(result.Status, LoadStatus::BadPadding) << offset;
        EXPECT_EQ(result.Tensor, 4                     ) << offset;
        EXPECT_EQ(result.Index , offset                ) << offset;
    }

    const auto mismatch = [&](const std::span<const std::byte> candidate) {
        const LoadResult result = network->Load(candidate);

        return result.Status == LoadStatus::SizeMismatch && result.Tensor == 4 && result.Index == candidate.size();
    };

    std::vector<std::byte> longer = bytes;

    for (s00 i = Padded; i < Padded + 64; i++) longer.push_back(static_cast<std::byte>("bullet"[(i - Payload) % 6]));

    EXPECT_TRUE(mismatch(std::span(bytes ).first(Payload - 1)));
    EXPECT_TRUE(mismatch(std::span(bytes ).first(Payload + 1)));
    EXPECT_TRUE(mismatch(std::span(bytes ).first(Padded  - 1)));
    EXPECT_TRUE(mismatch(std::span(bytes ).first(0          )));
    EXPECT_TRUE(mismatch(std::span(longer)                   ));
    EXPECT_TRUE(mismatch(std::span(longer).first(Padded  + 1)));

    EXPECT_EQ(Evaluations(*network, positions), expected);
}

TEST(Loader, BoundViolation)
{
    using namespace LoaderTest;

    std::mt19937_64 random(0x424F554E44535631);

    const std::vector<std::vector<u16>> positions = Positions(random);

    const std::vector<std::byte> bytes = RandomBytes<A>(random);

    const auto network = std::make_unique<Network<A>>();

    ASSERT_TRUE(network->Load(bytes));

    const std::vector<i32> expected = Evaluations(*network, positions);

    struct Case
    {

        s00 Offset;
        i16 Value ;
        u16 Tensor;
        u64 Index ;

    };

    constexpr std::array<Case, 6> Cases {{
        { L1 +  700 * 2,  128, 2,  700 },
        { L1 +    0 * 2, -128, 2,    0 },
        { L1 +  767 * 2, 1000, 2,  767 },
        { L0 + 4321 * 2,  506, 0, 4321 },
        { L0 +    0 * 2, -506, 0,    0 },
        { B0 +  383 * 2,  506, 1,  383 }
    }};

    for (const Case& violation : Cases) {
        std::vector<std::byte> patched = bytes;

        Patch(patched, violation.Offset, violation.Value);

        const LoadResult result = network->Load(patched);

        EXPECT_EQ(result.Status, LoadStatus::BoundViolation) << violation.Offset;
        EXPECT_EQ(result.Tensor, violation.Tensor          ) << violation.Offset;
        EXPECT_EQ(result.Index , violation.Index           ) << violation.Offset;
    }

    EXPECT_EQ(Evaluations(*network, positions), expected);

    std::vector<std::byte> bounds = bytes;

    Patch(bounds, L1 +  700 * 2, i16 {    127 });
    Patch(bounds, L1 +    0 * 2, i16 {   -127 });
    Patch(bounds, L0 + 4321 * 2, i16 {    505 });
    Patch(bounds, B0 +  383 * 2, i16 {   -505 });
    Patch(bounds, B1           , i16 { -32768 });

    EXPECT_TRUE(network->Load(bounds));
}

TEST(Loader, FailedLoadLeavesNetwork)
{
    using namespace LoaderTest;

    std::mt19937_64 random(0x4641494C4C4F4144);

    const std::vector<std::vector<u16>> positions = Positions(random);

    std::vector<std::byte> bytes = RandomBytes<A>(random);

    Pad(bytes);

    const auto network = std::make_unique<Network<A>>();

    ASSERT_TRUE(network->Load(bytes));

    const std::vector<std::byte> snapshot = Bytes(*network);
    const std::vector<i32>       expected = Evaluations(*network, positions);

    std::vector<std::byte> padding = bytes;
    std::vector<std::byte> late    = bytes;
    std::vector<std::byte> longer  = bytes;

    padding[Padded - 2] = std::byte { 'x' };

    Patch(late, L1 + 767 * 2, i16 { -300 });

    longer.resize(Padded + 64, std::byte { 'b' });

    const std::array<std::span<const std::byte>, 6> failures {
        std::span(bytes  ).first(Payload - 1),
        std::span(bytes  ).first(1000       ),
        std::span(bytes  ).first(0          ),
        std::span(longer ),
        std::span(padding),
        std::span(late   )
    };

    for (s00 i = 0; i < failures.size(); i++) {
        EXPECT_FALSE(network->Load(failures[i])) << "failure " << i;

        EXPECT_EQ(Bytes(*network), snapshot) << "failure " << i;
    }

    EXPECT_EQ(Evaluations(*network, positions), expected);
}

TEST(Loader, StageErrors)
{
    using M  = Examples::MultiLayer;
    using P  = Backend::Program<M, Backend::Native>;
    using FT = M::Transformer;

    constexpr s00 Transformer = (FT::Features::InputSize + 1) * FT::OutputSize * sizeof(i16);

    std::mt19937_64 random(0x5354414745455252);

    std::vector<std::byte> bytes = RandomBytes<M>(random);

    const auto parameters = std::make_unique<P::Parameters>();

    {
        IO::Reader reader(std::span(bytes).subspan(Transformer));

        EXPECT_TRUE(P::Load(reader, *parameters));
        EXPECT_EQ(reader.Remaining(), s00 { 0 });
        EXPECT_EQ(reader.Tensor   (), 6);
    }

    {
        IO::Reader reader(std::span(bytes).subspan(Transformer).first(bytes.size() - Transformer - 1));

        const LoadResult result = P::Load(reader, *parameters);

        EXPECT_EQ(result.Status, LoadStatus::SizeMismatch);
        EXPECT_EQ(result.Tensor, 5);
    }

    LoaderTest::Patch(bytes, Transformer + 5, i08 { -128 });

    {
        IO::Reader reader(std::span(bytes).subspan(Transformer));

        const LoadResult result = P::Load(reader, *parameters);

        EXPECT_EQ(result.Status, LoadStatus::BoundViolation);
        EXPECT_EQ(result.Tensor, 0);
        EXPECT_EQ(result.Index , u64 { 5 });
    }

    const LoadResult result = std::make_unique<Network<M>>()->Load(bytes);

    EXPECT_EQ(result.Status, LoadStatus::BoundViolation);
    EXPECT_EQ(result.Tensor, 2);
    EXPECT_EQ(result.Index , u64 { 5 });
}

TEST(Loader, NonFiniteValues)
{
    using M  = Examples::MultiLayer;
    using FT = M::Transformer;

    constexpr s00 B1  = (FT::Features::InputSize + 1) * FT::OutputSize * sizeof(i16) + 64 * 128 * sizeof(i08);
    constexpr s00 L2  = B1 + 128 * sizeof(f32);
    constexpr s00 B3  = L2 + (16 * 256 + 256 + 32 * 8) * sizeof(f32);
    constexpr s00 End = B3 + 8 * sizeof(f32);

    std::mt19937_64 random(0x4E4F4E46494E4954);

    const std::vector<std::vector<u16>> positions = LoaderTest::Positions(random, true);

    const std::vector<std::byte> bytes = RandomBytes<M>(random);

    ASSERT_EQ(bytes.size(), End);

    const auto network = std::make_unique<Network<M>>();

    ASSERT_TRUE(network->Load(bytes));

    const std::vector<std::byte> snapshot = LoaderTest::Bytes(*network);
    const std::vector<i32>       expected = LoaderTest::Evaluations(*network, positions);

    struct Case
    {

        s00 Offset;
        u32 Bits  ;
        u16 Tensor;
        u64 Index ;

    };

    constexpr std::array<Case, 4> Cases {{
        { B1 +   0 * 4, 0x7FC00000, 3,   0 },
        { B1 + 127 * 4, 0x7F800001, 3, 127 },
        { L2 + 100 * 4, 0x7F800000, 4, 100 },
        { B3 +   7 * 4, 0xFF800000, 7,   7 }
    }};

    for (const Case& value : Cases) {
        std::vector<std::byte> patched = bytes;

        LoaderTest::Patch(patched, value.Offset, value.Bits);

        const LoadResult result = network->Load(patched);

        EXPECT_EQ(result.Status   , LoadStatus::BoundViolation) << value.Offset;
        EXPECT_EQ(result.Tensor   , value.Tensor              ) << value.Offset;
        EXPECT_EQ(result.Index    , value.Index               ) << value.Offset;
        EXPECT_EQ(result.Message(), "bound violation"         ) << value.Offset;

        EXPECT_EQ(LoaderTest::Bytes(*network), snapshot) << value.Offset;
    }

    EXPECT_EQ(LoaderTest::Evaluations(*network, positions), expected);

    std::vector<std::byte> finite = bytes;

    LoaderTest::Patch(finite, B1 +   0 * 4, u32 { 0x7F7FFFFF });
    LoaderTest::Patch(finite, L2 + 100 * 4, u32 { 0xFF7FFFFF });
    LoaderTest::Patch(finite, B3 +   7 * 4, u32 { 0x00000001 });

    EXPECT_TRUE(network->Load(finite));
}

#endif //MANTARAY_IO_GTESTLOADER_H
