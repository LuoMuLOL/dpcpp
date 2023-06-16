#include "curve25519_donna.h"
#include <sycl/sycl.hpp>
#include <thread>

using namespace sycl;

//__attribute__((mode(TI)))是GCC编译器提供的一种扩展语法，用于指定数据类型的底层实现方式
//将uint128_t定义为一个无符号128位整数类型，并使用两个无符号64位整数来存储它的值。
typedef unsigned uint128_t __attribute__((mode(TI)));


queue q {cpu_selector_v};
//两个大小为5的无符号64位整型数组相加: output += in 
static inline void force_inline
fsum(limb *output, const limb *in) {
  buffer<limb, 1> output_buf{ output, range<1>{5} };
  buffer<const limb, 1> in_buf{ in, range<1>{5} };
  
  q.submit([&](handler &h){
    auto output_acc = output_buf.get_access<access::mode::read_write>(h);
    auto in_acc = in_buf.get_access<access::mode::read>(h);
    //对数组每个元素进行加法操作
    h.parallel_for(range<1>{5}, [=](id<1> i) {
      output_acc[i] += in_acc[i];
    });
  }).wait();
}

/* 两个不同的数之间的差: output = in - output
   执行前 out[i] < 2^52；执行后 out[i] < 2^55 */
static inline void force_inline
fdifference_backwards(felem out, const felem in) {
  buffer<limb, 1> out_buf{ out, range<1>{5} };
  buffer<const limb, 1> in_buf{ in, range<1>{5} };

  q.submit([&](handler &h){
    auto out_acc = out_buf.get_access<access::mode::read_write>(h);
    auto in_acc = in_buf.get_access<access::mode::read>(h);
    //确保结果的值域在 [0,2^55)之间
    constexpr limb two54m152 = static_cast<limb>((1UL << 54) - 152);  // 2^54−152 
    constexpr limb two54m8 = static_cast<limb>((1UL << 54) - 8);      // 2^54-8
    // 若两个数组中的元素直接相减，结果可能会溢出
    h.parallel_for(range<1>{5}, [=](id<1> i) {
       out_acc[i] = in_acc[i] + ((i == 0) ? two54m152 : two54m8) - out_acc[i];
    });
  }).wait();
}

//数组（in）乘以一个常量(scalar)，并将结果输出到output数组中: output = in * scalar 
static inline void force_inline
fscalar_product(felem output, const felem in, const limb scalar) {
  uint128_t a;
  uint64_t t[5][2];

  buffer<const limb, 1> inputBuf(in, range<1>{5});
  buffer<uint64_t, 2> t_Buf(reinterpret_cast<limb*>(t), range<2>{5,2});
  limb* scalar_shared=malloc_shared<limb>(sizeof(limb) ,q);
  memcpy(scalar_shared,&scalar,sizeof(uint64_t));
  //1.计算输入数组 in 中对应元素和标量 scalar 的乘积，并将结果存储在数组a中。
  //2.取出 a 的低 51 位作为输出数组的当前元素。
  //3.更新 a 的高 77 位（实际上就是右移 51 位后的剩余部分）以备下次迭代使用。
   q.submit([&](handler &h) {
    accessor inputData(inputBuf, h, read_only);
    accessor tData(t_Buf,h,write_only);
    //与多项式与一个常量相乘类似
    h.parallel_for(range<1>(5) , [=](id<1> i){
       tData[i][0] =inputData[i]*(*scalar_shared);         //低64位
       tData[i][1]=mul_hi(inputData[i],*scalar_shared);    //高64位
    });
   }).wait();
   host_accessor tData_host (t_Buf,read_only);
   free(scalar_shared,q);
   
   //规约处理，依次迭代(Barrett reduction规约方法)
   a = tData_host[0][0] | (static_cast<uint128_t>(tData_host[0][1]) << 64);
   output[0] = static_cast<uint64_t>(a) & 0x7ffffffffffff;  //取出低51位
   for (int i = 1; i < 5; i++) {
     a = (tData_host[i][0] | (static_cast<uint128_t>(tData_host[i][1]) << 64)) + (a >> 51);
     output[i] = static_cast<uint64_t>(a) & 0x7ffffffffffff;
  }
  
  //满足curve25519 曲线的 X 坐标要求，小于2^255-19
   output[0] += (a >> 51) * 19;
}

/* 两个数据相乘: output = in2 * in
 * 输出必须不同于两个输入；输入是规约系数形式，输出不是
 * 函数执行前参数 in[i] < 2^55 ，in2[i]也一样。
 * 执行后 output[i] < 2^52                           */
static inline void force_inline
fmul(felem output, const felem in2, const felem in) {
  uint128_t t[5];
  //看做是多项式系数
  //A(x) = a + bx + cx^2 + dx^3 + ex^4
  //B(x) = A + Bx + Cx^2 + Dx^3 + Ex^4

  // 两多项式乘积，根据多项式未知量的次方数区分
  // A(x)*B(x)=a*A + (a*B + b*A)*x + (a*C + b*B + c*A)*x^2 + (a*D + b*C + c*B + d*A)*x^3 + (a*E + b*D + c*C + d*B + e*A)*x^4 
  //          +(b*E + c*D + d*C + e*B)*x^5 + (c*E + d*D + e*C)*x^6 + (d*E + e*D)*x^7 + e*E*x^8
  limb a[5][5][2],c;
  buffer<limb,3> a_buf(reinterpret_cast<limb*>(a),range<3>{5,5,2});
  buffer<const limb,1> in_buf(in,range<1>{5});
  buffer<const limb,1> in2_buf(in2,range<1>{5});

  auto eA= q.submit([&](handler& h){
     auto a_acc = a_buf.get_access<access::mode::write>(h);
     auto in_acc = in_buf.get_access<access::mode::read>(h);
     auto in2_acc = in2_buf.get_access<access::mode::read>(h);
     h.parallel_for(range<2>{5, 5}, [=](id<2> idx){
        int i = idx[0];
        int j = idx[1];
        if (j < 5 - i) {
            a_acc[i][j][0] = in_acc[i] * in2_acc[j];
            a_acc[i][j][1] = mul_hi(in_acc[i], in2_acc[j]);
        }
    });
  });
 
 //19 是 Curve25519 算法中使用的固定值，用于将结果缩小到正确范围内 
 //多项式乘积的次方数大于4时要进行模运算（超出curv25519曲线规定的值)，结果缩小到正确的取模范围内
  auto eB=q.submit([&](handler& h){
     h.depends_on(eA);
     auto a_acc = a_buf.get_access<access::mode::write>(h);
     auto in_acc = in_buf.get_access<access::mode::read>(h);
     auto in2_acc = in2_buf.get_access<access::mode::read>(h);

     h.parallel_for(range<2>{4, 4}, [=](id<2> idx){
         int i = 4 - idx[0];
         int j = 4 - idx[1];
         if (j >= 5 - i ) {
            a_acc[i][j][0] = in_acc[i]* 19 * in2_acc[j];
            a_acc[i][j][1] = mul_hi(in_acc[i]*19, in2_acc[j]);
        }
    });
  });
   eB.wait();
   host_accessor aData(a_buf,read_only);
   //合并同类项
   t[0]=( ((uint128_t)aData[0][0][1]) <<64) | aData[0][0][0];
   t[1]=( (((uint128_t)aData[0][1][1]) <<64)| aData[0][1][0])+( (((uint128_t)aData[1][0][1]) <<64) | aData[1][0][0] );
   t[2]=( (((uint128_t)aData[0][2][1]) <<64) | aData[0][2][0])+( (((uint128_t)aData[1][1][1]) <<64) | aData[1][1][0])+( (((uint128_t)aData[2][0][1]) <<64) | aData[2][0][0]);
   t[3]=( (((uint128_t)aData[0][3][1]) <<64) | aData[0][3][0])+( (((uint128_t)aData[1][2][1]) <<64) | aData[1][2][0])+( (((uint128_t)aData[2][1][1]) <<64) | aData[2][1][0])+( (((uint128_t)aData[3][0][1]) <<64) | aData[3][0][0]);
   t[4]=( (((uint128_t)aData[0][4][1]) <<64) | aData[0][4][0])+( (((uint128_t)aData[1][3][1]) <<64) | aData[1][3][0])+( (((uint128_t)aData[2][2][1]) <<64) | aData[2][2][0])+( (((uint128_t)aData[3][1][1]) <<64) | aData[3][1][0])+( (((uint128_t)aData[4][0][1]) <<64) | aData[4][0][0]);
 
   t[0]+=( (((uint128_t)aData[4][1][1]) <<64) | aData[4][1][0])+( (((uint128_t)aData[3][2][1]) <<64) | aData[3][2][0])+( (((uint128_t)aData[2][3][1]) <<64) | aData[2][3][0])+( (((uint128_t)aData[1][4][1]) <<64) | aData[1][4][0]);
   t[1]+=( (((uint128_t)aData[4][2][1]) <<64) | aData[4][2][0])+( (((uint128_t)aData[3][3][1]) <<64) | aData[3][3][0])+( (((uint128_t)aData[2][4][1]) <<64) | aData[2][4][0]);
   t[2]+=( (((uint128_t)aData[4][3][1]) <<64) | aData[4][3][0])+( (((uint128_t)aData[3][4][1]) <<64) | aData[3][4][0]);
   t[3]+=( ((uint128_t)aData[4][4][1]) <<64) | aData[4][4][0];

      
  //将中间值对 2^51 取模，得到低 51 位的值作为多项式的系数。
  //将中间值右移 51 位，得到进位值 c。
  //将进位值加到下一个中间结果上。
                  output[0] = (limb)t[0] & 0x7ffffffffffff; c = (limb)(t[0] >> 51);
  t[1] += c;      output[1] = (limb)t[1] & 0x7ffffffffffff; c = (limb)(t[1] >> 51);
  t[2] += c;      output[2] = (limb)t[2] & 0x7ffffffffffff; c = (limb)(t[2] >> 51);
  t[3] += c;      output[3] = (limb)t[3] & 0x7ffffffffffff; c = (limb)(t[3] >> 51);
  t[4] += c;      output[4] = (limb)t[4] & 0x7ffffffffffff; c = (limb)(t[4] >> 51);
  
  //最高位的进位值乘以 19 加到最低位上,确保结果仍然在取模范围内
  output[0] +=   c * 19; c = output[0] >> 51; output[0] = output[0] & 0x7ffffffffffff;
  output[1] +=   c;      c = output[1] >> 51; output[1] = output[1] & 0x7ffffffffffff;
  output[2] +=   c;
  
}

//求in的平方的count次方的结果:（in^2)^count
static inline void force_inline
fsquare_times(felem output, const felem in, limb count) {
  uint128_t t[5];
  // 多项式乘积
  // (a + bx + cx^2 + dx^3 + ex^4)^2 = a^2 + 2abx + (2ac + b^2)x^2 + (2ad + 2bc)x^3 + (2ae + 2bd + c^2)x^4 
  //                                 + (2cd + 2be)x^5 + (2ce + d^2)x^6 + 2ed x^7 + e^2x^8

  limb a[5][5][2],c;
  limb in_init[5] = {0};
  memcpy(in_init, in, sizeof(limb) * 5);
  buffer<limb,3> a_buf(reinterpret_cast<limb*>(a),range<3>{5,5,2});
  buffer<limb,1> in_buf(in_init, range<1>{5});

  do {
   auto eA= q.submit([&](handler& h){
       auto a_acc = a_buf.get_access<access::mode::write>(h);
       auto in_acc = in_buf.get_access<access::mode::read>(h);
      
       h.parallel_for(range<2>{5, 5}, [=](id<2> idx){
         int i = idx[0];
         int j = idx[1];
         if( i <= j ){
            if( i+j<5 ){
              if( i==j){
                  a_acc[i][j][0] = in_acc[i] * in_acc[j];
                  a_acc[i][j][1] = mul_hi(in_acc[i], in_acc[j]);
               }else if( i!=j ){
                  a_acc[i][j][0] = in_acc[i] *2 * in_acc[j];
                  a_acc[i][j][1] = mul_hi(in_acc[i]*2, in_acc[j]);
               }
             }else if( i+j>=5){
                if( i==j){
                  a_acc[i][j][0] = in_acc[i] *19 * in_acc[j];
                  a_acc[i][j][1] = mul_hi(in_acc[i]*19, in_acc[j]);
                }else if( i!=j ){
                  a_acc[i][j][0] = in_acc[i] *2 *19 * in_acc[j];
                  a_acc[i][j][1] = mul_hi(in_acc[i]*2*19, in_acc[j]);
               }
             }
           }
         });
     });
     eA.wait();
     host_accessor aData(a_buf,read_only);
     host_accessor inData(in_buf,read_write);
     //合并同类项
     t[0]=((((uint128_t)aData[0][0][1])<<64) |aData[0][0][0])+((((uint128_t)aData[1][4][1])<<64) |aData[1][4][0])+((((uint128_t)aData[2][3][1])<<64) |aData[2][3][0]);
     t[1]=((((uint128_t)aData[0][1][1])<<64) |aData[0][1][0])+((((uint128_t)aData[2][4][1])<<64) |aData[2][4][0])+((((uint128_t)aData[3][3][1])<<64) |aData[3][3][0]);
     t[2]=((((uint128_t)aData[0][2][1])<<64) |aData[0][2][0])+((((uint128_t)aData[1][1][1])<<64) |aData[1][1][0])+((((uint128_t)aData[3][4][1])<<64) |aData[3][4][0]);
     t[3]=((((uint128_t)aData[0][3][1])<<64) |aData[0][3][0])+((((uint128_t)aData[1][2][1])<<64) |aData[1][2][0])+((((uint128_t)aData[4][4][1])<<64) |aData[4][4][0]);
     t[4]=((((uint128_t)aData[0][4][1])<<64) |aData[0][4][0])+((((uint128_t)aData[1][3][1])<<64) |aData[1][3][0])+((((uint128_t)aData[2][2][1])<<64) |aData[2][2][0]);
   
    //拆分规约计算
    //确保了进位标志变量 c 的值在正确的范围内，并且可以被正确地用于后续计算
                    inData[0] = (limb)t[0] & 0x7ffffffffffff; c = (limb)(t[0] >> 51); 
    t[1] += c;      inData[1] = (limb)t[1] & 0x7ffffffffffff; c = (limb)(t[1] >> 51);
    t[2] += c;      inData[2] = (limb)t[2] & 0x7ffffffffffff; c = (limb)(t[2] >> 51);
    t[3] += c;      inData[3] = (limb)t[3] & 0x7ffffffffffff; c = (limb)(t[3] >> 51);
    t[4] += c;      inData[4] = (limb)t[4] & 0x7ffffffffffff; c = (limb)(t[4] >> 51);
    inData[0] +=   c * 19; c = inData[0] >> 51; inData[0] = inData[0] & 0x7ffffffffffff;
    inData[1] +=   c;      c = inData[1] >> 51; inData[1] = inData[1] & 0x7ffffffffffff;
    inData[2] +=   c;
  
  } while(--count);

  host_accessor inData(in_buf,read_only);

  memcpy(output,inData.get_pointer(),sizeof(limb)*5);
}


//将大小为8的8位无符号整数数组转换为64位无符号整数
static limb load_limb(const u8 *in) {
  return
     ((limb)in[0]) |
     (((limb)in[1]) << 8) |
     (((limb)in[2]) << 16) |
     (((limb)in[3]) << 24) |
     (((limb)in[4]) << 32) |
     (((limb)in[5]) << 40) |
     (((limb)in[6]) << 48) |
     (((limb)in[7]) << 56) ;
}

//将64位无符号整数存储到uint8_t数组中
static void store_limb(u8 *out, limb in) {
    buffer<u8, 1> outBuf(out, range<1>(sizeof(limb)));
    limb * indata=malloc_shared<limb>(sizeof(limb),q);
    memcpy(indata,&in,sizeof(limb));

    q.submit([&](handler &h) {
        accessor outData(outBuf, h, write_only);
      h.parallel_for(range<1>(8), [=](id<1> i) {
        outData[i] = static_cast<u8>(((*indata) >> (8 * i))) & 0xff;
      });
    }).wait();
    free(indata,q);
}

//将大小为32的uint8_t数组转换成大小为5的uint64_t数组
static void fexpand(limb *output, const u8 *in) {
   buffer<limb, 1> outBuf(output, range<1>(5));
   u8* indata=malloc_shared<u8>(32,q);
   memcpy(indata,in,sizeof(u8)*32);

   q.submit([&](handler &h) {
      accessor outData(outBuf, h, write_only);
      h.parallel_for(range<1>{5}, [=](id<1> idx){
         int i=idx[0];
         //每个数组元素存储51位无符号整数
         switch(i){
           case 0:outData[0] = load_limb(indata) & 0x7ffffffffffff;break;
           case 1:outData[1] = (load_limb(indata+6) >> 3) & 0x7ffffffffffff;break;
           case 2:outData[2] = (load_limb(indata+12) >> 6) & 0x7ffffffffffff;break;
           case 3:outData[3] = (load_limb(indata+19) >> 1) & 0x7ffffffffffff;break;
           case 4:outData[4] = (load_limb(indata+24) >> 12) & 0x7ffffffffffff;break;
        }
    });
  }).wait();
  free(indata,q);
}


// 将一个完全归约的多项式形式的数据，转换为一个大小为32的uint8_t数组(小端序)。
// 总体而言，这段代码主要是对多项式数据进行压缩，以便于在数据传输中使用，同时保证了数据的归约和进位。
static void
fcontract(u8 *output, const felem input) {
  uint128_t t[5];

  t[0] = input[0];
  t[1] = input[1];
  t[2] = input[2];
  t[3] = input[3];
  t[4] = input[4];

  t[1] += t[0] >> 51; t[0] &= 0x7ffffffffffff;
  t[2] += t[1] >> 51; t[1] &= 0x7ffffffffffff;
  t[3] += t[2] >> 51; t[2] &= 0x7ffffffffffff;
  t[4] += t[3] >> 51; t[3] &= 0x7ffffffffffff;
  t[0] += 19 * (t[4] >> 51); t[4] &= 0x7ffffffffffff;


  t[1] += t[0] >> 51; t[0] &= 0x7ffffffffffff;
  t[2] += t[1] >> 51; t[1] &= 0x7ffffffffffff;
  t[3] += t[2] >> 51; t[2] &= 0x7ffffffffffff;
  t[4] += t[3] >> 51; t[3] &= 0x7ffffffffffff;
  t[0] += 19 * (t[4] >> 51); t[4] &= 0x7ffffffffffff;

  /* 现在t的值在 0 到 2^255-1 之间，并且已经正确进位 */
  /* 情况1：t在0到2^255-20之间；情况2：t在2^255-19和2^255-1之间 */

  t[0] += 19;

  t[1] += t[0] >> 51; t[0] &= 0x7ffffffffffff;
  t[2] += t[1] >> 51; t[1] &= 0x7ffffffffffff;
  t[3] += t[2] >> 51; t[2] &= 0x7ffffffffffff;
  t[4] += t[3] >> 51; t[3] &= 0x7ffffffffffff;
  t[0] += 19 * (t[4] >> 51); t[4] &= 0x7ffffffffffff;

  /*现在在两种情况下t均在19和2^255-1之间，并且都偏移了19 */

  t[0] += 0x8000000000000 - 19;
  t[1] += 0x8000000000000 - 1;
  t[2] += 0x8000000000000 - 1;
  t[3] += 0x8000000000000 - 1;
  t[4] += 0x8000000000000 - 1;

  /* 现在t在 2^255 和 2^256-20 之间，并且都偏移了 2^255 */

  t[1] += t[0] >> 51; t[0] &= 0x7ffffffffffff;
  t[2] += t[1] >> 51; t[1] &= 0x7ffffffffffff;
  t[3] += t[2] >> 51; t[2] &= 0x7ffffffffffff;
  t[4] += t[3] >> 51; t[3] &= 0x7ffffffffffff;
  t[4] &= 0x7ffffffffffff;
  
  store_limb(output ,   t[0] | (t[1] << 51));
  store_limb(output+8,  (t[1] >> 13) | (t[2] << 38));
  store_limb(output+16, (t[2] >> 26) | (t[3] << 25));
  store_limb(output+24, (t[3] >> 39) | (t[4] << 12));
}


//Q坐标变换
void fmonty_task1(limb *x, limb *z, limb *origx) {
  memcpy(origx, x, 5 * sizeof(limb));
  fsum(x, z);                        // x = x + z     
  fdifference_backwards(z, origx);   // z = z - origx
}

//Q'坐标变换
void fmonty_task2(limb *xprime, limb *zprime, limb *origxprime) {
  memcpy(origxprime, xprime, sizeof(limb) * 5);
  fsum(xprime, zprime);                       //xprime = xprime + zprime
  fdifference_backwards(zprime, origxprime);  //zprime = zprime - origxprime
}

void fmonty_task3(limb *xxprime, limb *zzprime, limb *x, limb *z,limb *xprime,limb *zprime,limb *origxprime){
      fmul(xxprime, xprime, z);  // xxprime = xprime*z
      fmul(zzprime, x, zprime);  // zzprime = x*zprime

      memcpy(origxprime, xxprime, sizeof(limb) * 5);
      fsum(xxprime, zzprime);                      // xxprime = xxprime + zzprime
      fdifference_backwards(zzprime, origxprime);  // zzprime = origxprime - zzprime
}

//Q+Q'(x3 ，z3)
void fmonty_task4(limb *x3, limb *z3, const limb *qmqp, limb *xxprime, limb *zzprime, limb *zzzprime) {
  fsquare_times(x3, xxprime, 1);    // x3=xxprime ^2
  fsquare_times(zzzprime, zzprime, 1);   // zzzprime=zzprime ^2
  fmul(z3, zzzprime, qmqp);     // z3=zzzprime*qmqp
}

//2Q (x2 ，z2)
void fmonty_task5(limb *x2, limb *z2, limb *x,limb *z,limb *xx, limb *zz, limb *zzz) {
  fsquare_times(xx, x, 1);   // xx = x^2
  fsquare_times(zz, z, 1);   // zz = z^2
  fmul(x2, xx, zz);          // x2 = xx*zz

  fdifference_backwards(zz, xx);     // zz = xx - zz
  fscalar_product(zzz, zz, 121665);  // zzz = zz*121665
  fsum(zzz, xx);       // zzz = zzz+xx
  fmul(z2, zz, zzz);   // z2 = zz*zzz
}

/* 输入: Q, Q', Q-Q'
 * 输出: 2Q, Q+Q'  */
//蒙哥马利点乘计算
static void
fmonty(limb *x2, limb *z2,     //  2Q 
       limb *x3, limb *z3,     // Q + Q' 
       limb *x, limb *z,       // Q 
       limb *xprime, limb *zprime, // Q' 
       const limb *qmqp        /* Q - Q' */
       ) {
  limb origx[5], origxprime[5], zzz[5], xx[5], zz[5], xxprime[5],
        zzprime[5], zzzprime[5];
      //Q
      std::thread t1(fmonty_task1, x, z, origx);   // 启动 Task1 线程
      //Q'
      std::thread t2(fmonty_task2, xprime, zprime, origxprime); 
      
      t1.join(); // 等待 Task1 完成
      t2.join(); 
 
      std::thread t3(fmonty_task3, xxprime, zzprime, x, z, xprime, zprime ,origxprime);    
      t3.join();
      //Q+Q'
      std::thread t4(fmonty_task4, x3, z3, qmqp,xxprime,zzprime,zzzprime);   
      //2Q
      std::thread t5(fmonty_task5, x2, z2, x,z,xx,zz,zzz); 
      t4.join();
      t5.join();   
}

// 可能会交换两个长度为 5 的 limb 数组 a 和 b 的内容
// 当且仅当 iswap 非零时才执行交换操作
// 防止侧信道泄漏信息
static void swap_conditional(limb a[5], limb b[5], limb iswap) {
   buffer<limb, 1> a_buf{a, range<1>{5}};
   buffer<limb, 1> b_buf{b, range<1>{5}};
   limb* iswap_usm = malloc_shared<limb>(1, q);
   memcpy(iswap_usm,&iswap,sizeof(limb));
   
   q.submit([&](handler &h){
        accessor a_acc{a_buf, h};
        accessor b_acc{b_buf, h}; 
    h.parallel_for(range<1>(5), [=] (id<1> i) {
       const limb x = (-(*iswap_usm)) & (a_acc[i] ^ b_acc[i]);
       a_acc[i] ^= x;
       b_acc[i] ^= x;
   });
  }).wait();
   free(iswap_usm,q);
}

/* 计算曲线上一点Q的n倍点nQ，其中Q的x坐标已知
 *   resultx/resultz: 结果点的x坐标
 *   n: 一个小端序的32字节数字 
 *   q: 曲线上的一点     */
// 改进的double-and-add 算法计算公钥
static void
cmult(limb *resultx, limb *resultz, const u8 *n, const limb *q) {
  limb a[5] = {0}, b[5] = {1}, c[5] = {1}, d[5] = {0};
  limb *nqpqx = a, *nqpqz = b, *nqx = c, *nqz = d, *t;
  limb e[5] = {0}, f[5] = {1}, g[5] = {0}, h[5] = {1};
  limb *nqpqx2 = e, *nqpqz2 = f, *nqx2 = g, *nqz2 = h;
  unsigned i, j;

  memcpy(nqpqx, q, sizeof(limb) * 5);
  for (i = 0; i < 32; ++i) {
    u8 byte = n[i];
    for (j = 0; j < 8; ++j) {
      const limb bit = byte >> 7;
      swap_conditional(nqx, nqpqx, bit);   //避免了使用条件分支
      swap_conditional(nqz, nqpqz, bit);
      fmonty(nqx2, nqz2,      
             nqpqx2, nqpqz2,  
             nqx, nqz,        //R0
             nqpqx, nqpqz,    //R1
             q);              
      swap_conditional(nqx2, nqpqx2, bit);
      swap_conditional(nqz2, nqpqz2, bit);
      t = nqx;
      nqx = nqx2;
      nqx2 = t;
      
      t = nqz;
      nqz = nqz2;
      nqz2 = t;
      
      t = nqpqx;
      nqpqx = nqpqx2;
      nqpqx2 = t;
      
      t = nqpqz;
      nqpqz = nqpqz2;
      nqpqz2 = t;

      byte <<= 1;
    }
  }
  memcpy(resultx, nqx, sizeof(limb) * 5);
  memcpy(resultz, nqz, sizeof(limb) * 5);
}

//求有限域上z的逆元(扩展欧几里得算法)
static void crecip(felem out, const felem z) {
   felem a,t0,b,c;
   //通过一系列乘法和平方运算
   fsquare_times(a, z, 1); 
   fsquare_times(t0, a, 2); 
   fmul(b, t0, z); 
   fmul(a, b, a); 
   fsquare_times(t0, a, 1);
   fmul(b, t0, b);
   fsquare_times(t0, b, 5);
   fmul(b, t0, b);
   fsquare_times(t0, b, 10);
   fmul(c, t0, b);
   fsquare_times(t0, c, 20);
   fmul(t0, t0, c);
   fsquare_times(t0, t0, 10);
   fmul(b, t0, b);
   fsquare_times(t0, b, 50);
   fmul(c, t0, b);
   fsquare_times(t0, c, 100);
   fmul(t0, t0, c);
   fsquare_times(t0, t0, 50);
   fmul(t0, t0, b);
   fsquare_times(t0, t0, 5);
   fmul(out, t0, a);
}

//计算公钥
int curve25519_donna(u8 *mypublic, const u8 *secret, const u8 *basepoint) {
    limb bp[5], x[5], z[5], zmone[5];
    uint8_t e[32];
    
    memcpy(e,secret,sizeof(u8)*32);
    
    e[0] &= 248;
    e[31] &= 127;
    e[31] |= 64;

    fexpand(bp, basepoint);
    cmult(x, z, e, bp);
    crecip(zmone, z);
    fmul(z, x, zmone);  //将 x 转换成椭圆曲线有限域上的元素
    fcontract(mypublic, z);
  return 0;
}

/* 测试样例1
 * 该函数可以用于测试代码是否能正确处理非规范曲线点（即设置了第256位的点）。
 * 在某些情况下，可能会出现设置了第256位的点，这种点不能被视为有效的曲线点，
 * 因此需要进行特殊处理，以保证运算结果的准确性和安全性。                  */
int test1(){
    static const uint8_t point1[32] = {
    0x25,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
  }; //规范点
  static const uint8_t point2[32] = {
    0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,
    0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,
    0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,
    0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,
  }; //非规范点
  static const uint8_t scalar[32] = { 1 };
  uint8_t out1[32], out2[32];
  //计算公钥
  curve25519_donna(out1, scalar, point1);
  curve25519_donna(out2, scalar, point2);
  //若相等，则未被正确处理
  if(memcmp(out1, out2, sizeof(out1))== 0) {
     fprintf(stderr, "非规范曲线点最高位没有被忽略。\n");
     return 1;
  }
  fprintf(stderr, "非规范曲线点最高位被正确处理。\n");
  return 0;
}

//测试样例2
void doit(unsigned char *ek,unsigned char *e,unsigned char *k)
{
  //输出基点，公钥，共享密钥
  int i;
  for (i = 0;i < 32;++i) printf("%02x",(unsigned int) e[i]); printf(" ");
  for (i = 0;i < 32;++i) printf("%02x",(unsigned int) k[i]); printf(" ");
  curve25519_donna(ek,e,k);
  for (i = 0;i < 32;++i) printf("%02x",(unsigned int) ek[i]); printf("\n");
}
//验证共享密钥以及公钥的生成
int test2(){
  unsigned char e1k[32]; //公钥
  unsigned char e2k[32];
  unsigned char e1e2k[32]; //共享密钥
  unsigned char e2e1k[32];
  unsigned char e1[32] = {3};
  unsigned char e2[32] = {5};
  unsigned char k[32] = {9};

  for (int loop = 0;loop < 10;++loop) {
    doit(e1k,e1,k);
    doit(e2k,e2,k);

    doit(e2e1k,e2,e1k);
    doit(e1e2k,e1,e2k);
    for (int i = 0;i < 32;++i) 
      if(e1e2k[i] != e2e1k[i]) {
        printf("计算共享密钥有误\n");
        return 1;
      }
    for (int i = 0;i < 32;++i) e1[i] ^= e2k[i];  //异或
    for (int i = 0;i < 32;++i) e2[i] ^= e1k[i];
    for (int i = 0;i < 32;++i) k[i] ^= e1e2k[i];
  }
  return 0;  
}

//测试样例3
static uint64_t
time_now() {
  struct timeval tv;
  uint64_t ret;
  gettimeofday(&tv, NULL);  //获取当前时间
  ret = tv.tv_sec;    //秒
  ret *= 1000000;     //转换成微秒
  ret += tv.tv_usec;  //加上微秒

  return ret;
}
//测运行速度
void test3(){
  static const unsigned char basepoint[32] = {9};
  unsigned char mysecret[32], mypublic[32];
  unsigned i;
  uint64_t start, end;

  memset(mysecret, 42, 32);
  mysecret[0] &= 248;
  mysecret[31] &= 127;
  mysecret[31] |= 64;
  //预先加载缓存
  for (i = 0; i < 10; ++i) {
    curve25519_donna(mypublic, mysecret, basepoint);
  }
  start = time_now();
  for (i = 0; i < 10; ++i) {
    curve25519_donna(mypublic, mysecret, basepoint);
  }
  end = time_now();
  printf("%luus\n", (unsigned long) ((end - start) / 30000));
  return;
}