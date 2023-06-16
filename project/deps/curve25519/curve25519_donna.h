#pragma once

#include <cstring>
#include <cstdint>


typedef uint8_t u8;
typedef uint64_t limb;
typedef limb felem[5];
#undef force_inline
#define force_inline __attribute__((always_inline))

static inline void force_inline fsum(limb *output, const limb *in);
static inline void force_inline fdifference_backwards(felem out, const felem in);
static inline void force_inline fscalar_product(felem output, const felem in, const limb scalar);
static inline void force_inline fmul(felem output, const felem in2, const felem in);
static inline void force_inline fsquare_times(felem output, const felem in, limb count);
static limb load_limb(const u8 *in);
static void store_limb(u8 *out, limb in);
static void fexpand(limb *output, const u8 *in);
static void fcontract(u8 *output, const felem input);
static void fmonty(limb *x2,limb *z2,limb *x3,limb *z3,limb *x,limb *z,limb *xprime,limb *zprime,const limb *qmqp);
static void swap_conditional(limb a[5], limb b[5], limb iswap);
static void cmult(limb *resultx, limb *resultz, const u8 *n, const limb *q);
static void crecip(felem out, const felem z);
int curve25519_donna(u8 *mypublic, const u8 *secret, const u8 *basepoint);
int test1();
int test2();
void test3();