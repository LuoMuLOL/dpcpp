#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <cstring>
#include <arpa/inet.h>
#include <iostream>
#include <sodium.h>
#include "curve25519_donna.h"

#define MESSAGE_LEN 1024
const uint8_t BASE_POINT[32] = {9};  //curve25519曲线上的基点x坐标

int main()
{    
    //初始化libsodium
    if (sodium_init() != 0) {
      std::cerr << "初始化libsodium失败" << std::endl;
    }
    //创建通信的套接字
    int fd = socket(AF_INET, SOCK_STREAM, 0);  //ipv4 ; TCP协议
    if(fd == -1)
    {
       perror("socket");
       exit(0);
    }

    //连接服务器
    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(10000);    //大端端口
    inet_pton(AF_INET, "172.17.139.170", &addr.sin_addr.s_addr);  //将ipv4地址转换成大端序
    //向指定的服务器地址和端口号发起连接请求。
    int ret = connect(fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr));
    if(ret == -1)
    {
        perror("connect");
        exit(0);
    }
    
    uint8_t remote_public_key[crypto_scalarmult_curve25519_BYTES];
    uint8_t local_public_key[crypto_scalarmult_curve25519_BYTES];
    uint8_t remote_private_key[crypto_scalarmult_curve25519_SCALARBYTES];
    uint8_t shared_secret2[crypto_scalarmult_curve25519_BYTES];
    uint8_t shared_secret1[crypto_scalarmult_curve25519_BYTES];
    randombytes_buf(remote_private_key, sizeof remote_private_key);  //随机私钥

    if(curve25519_donna(remote_public_key,remote_private_key,BASE_POINT)!=0){
       std::cerr << "计算本地公钥失败" << std::endl;
       return -1;
    }
   
    std::cout << "Bob私钥: ";
    for (size_t i = 0; i < crypto_scalarmult_curve25519_SCALARBYTES; ++i) {
      std::printf("%02x", remote_private_key[i]);
    }
    std::cout << std::endl;
    std::cout << "Bob公钥: ";
    for (size_t i = 0; i < crypto_scalarmult_curve25519_BYTES; ++i) {
      std::printf("%02x", remote_public_key[i]);
    }
    std::cout << std::endl;
    //和服务器端通信
      //向Alice发送公钥数据
      send(fd, remote_public_key, crypto_scalarmult_curve25519_BYTES,0);
        
      //接收服务器公钥数据
      int len = recv(fd, local_public_key, sizeof(local_public_key),0);
      if(len > 0){   
          std::cout <<"Alice公钥: ";
          for(size_t i = 0; i < crypto_scalarmult_curve25519_BYTES; ++i) {
              std::printf("%02x", local_public_key[i]);
          }
          std::cout << std::endl;
      }
      else if(len  == 0)
      {
        printf("Alice端断开了连接...\n");
      }
      else { perror("recv");}

    //计算共享密钥
    if(curve25519_donna(shared_secret2,remote_private_key,local_public_key)!=0){
       std::cerr << "计算远程公钥失败" << std::endl;
       return -1;
    }
    //打印共享密钥
    std::cout << "共享密钥: ";
    for (size_t i = 0; i < crypto_scalarmult_curve25519_BYTES; ++i) {
         std::printf("%02x", shared_secret2[i]);
    }
    std::cout << std::endl;
      //发送共享密钥
      send(fd, shared_secret2, crypto_scalarmult_curve25519_BYTES,0);  
      //接收共享密钥
      len = recv(fd, shared_secret1, sizeof(shared_secret1),0);
      if(len > 0)
      {   
        // 比较两个共享密钥是否相同
        if (memcmp(shared_secret1, shared_secret2, crypto_scalarmult_curve25519_BYTES)!= 0) {
             std::cerr << "共享密钥不匹配。" << std::endl;
             return -1;
         }else {
             std::cout<<"共享密钥匹配"<<std::endl;
         }
      }
      else if(len  == 0)
      {
        printf("Alice端断开了连接...\n");
      }
      else { perror("recv"); }
     uint8_t nonce[crypto_box_NONCEBYTES];
     unsigned char decrypted_text[MESSAGE_LEN];      
     unsigned char cipher_text[MESSAGE_LEN + crypto_box_MACBYTES];

     len=recv(fd,cipher_text,MESSAGE_LEN + crypto_box_MACBYTES,0);
     if(len<=0){ std::cout<<"传输失败"<<std::endl; }
     len=recv(fd,nonce,crypto_box_NONCEBYTES,0);
     if(len<=0){ std::cout<<"传输失败"<<std::endl; }
     //解密信息
     if(crypto_box_open_easy_afternm(decrypted_text, cipher_text, sizeof(cipher_text), nonce,shared_secret2)!=0){
      std::cerr << "解密信息失败。" << std::endl;
      return -1;
     }
    std::cout << "解密后的message: " << decrypted_text << std::endl;
    
    close(fd);
    return 0;
}
