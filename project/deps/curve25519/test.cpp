#include "curve25519_donna.h"
#include <iostream>

//测试代码
int main(){
  if(test1()==1){     //测试非规则点
     std::cerr<<"椭圆曲线加密算法有误"<<std::endl;
     return -1;
   }

   if(test2()==1){    //测试计算公钥过程无误
     std::cerr<<"椭圆曲线加密算法有误"<<std::endl;
     return -1;
   }

   test3();     //测试运行速度
   return 0;
}