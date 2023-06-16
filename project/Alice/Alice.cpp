#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <cstring>
#include <arpa/inet.h>
#include <sodium.h>
#include <iostream>
#include "curve25519_donna.h"

#define MESSAGE_LEN 1024   //加密数据大小
const uint8_t BASE_POINT[32] = {9};  //curve25519曲线上的基点x坐标

int main()
{     
    //初始化libsodium
    if(sodium_init() != 0) {
      std::cerr << "初始化libsodium失败" << std::endl;
      return -1;
    }
   
    //创建监听的套接字
    int lfd = socket(AF_INET, SOCK_STREAM, 0); //支持IPv4协议、面向流（TCP）传输的套接字
    if(lfd == -1)
    {
        perror("socket");
        exit(0);
    }

    //将套接字和本地的IP端口绑定到一起
    struct sockaddr_in saddr;
    saddr.sin_family = AF_INET;
    saddr.sin_port = htons(10000);     //端口，将主机字节序的端口号转换为网络字节序
    //INADDR_ANY宏的值为0 == 0.0.0.0,可以代表任意一个IP地址，自动寻找符合IP地址绑定
    saddr.sin_addr.s_addr = INADDR_ANY;  
    int ret = bind(lfd, reinterpret_cast<struct sockaddr*>(&saddr), sizeof(saddr));
    if(ret == -1)
    {
        perror("bind");
        exit(0);
    }
    //设置套接字为监听状态，以便接受客户端的连接请求
    ret = listen(lfd, 128);
    if(ret == -1)
    {
        perror("listen");
        exit(0);
    }
    //阻塞等待并接受客户端连接
    struct sockaddr_in cliaddr;        //传出参数
    socklen_t clilen = sizeof(cliaddr);
    int cfd = accept(lfd, reinterpret_cast<struct sockaddr*>(&cliaddr), &clilen); 
    if(cfd == -1)
    {
        perror("accept");
        exit(0);
    }
    //打印客户端的地址信息
    char ip[24] = {0};
    printf("客户端的IP地址: %s, 端口: %d\n",
           inet_ntop(AF_INET, &cliaddr.sin_addr.s_addr, ip, sizeof(ip)),   //将IP地址转换为点分十进制格式
           ntohs(cliaddr.sin_port));              //将网络字节序的端口号转换为主机字节序的端口号

    //和客户端通信
    uint8_t local_public_key[crypto_scalarmult_curve25519_BYTES];
    uint8_t remote_public_key[crypto_scalarmult_curve25519_BYTES];
    uint8_t local_private_key[crypto_scalarmult_curve25519_SCALARBYTES];
    uint8_t shared_secret1[crypto_scalarmult_curve25519_BYTES];
    uint8_t shared_secret2[crypto_scalarmult_curve25519_BYTES];
    randombytes_buf(local_private_key, sizeof(local_private_key));    //随机生成私钥

    if(curve25519_donna(local_public_key,local_private_key,BASE_POINT)!=0){
       std::cerr << "计算本地公钥失败" << std::endl;
       return -1;
    }
    //打印密钥对
    std::cout << "Alice私钥: ";
    for (size_t i = 0; i < crypto_scalarmult_curve25519_SCALARBYTES; ++i) {
      std::printf("%02x", local_private_key[i]);
    }
    std::cout << std::endl;
    std::cout << "Alice公钥: ";
    for (size_t i = 0; i < crypto_scalarmult_curve25519_BYTES; ++i) {
      std::printf("%02x", local_public_key[i]);
    }
    std::cout << std::endl;

    //接收客户端数据
    int len = recv(cfd, remote_public_key, sizeof(remote_public_key),0);
    if(len > 0)
    {
        std::cout << "Bob公钥: ";
        for (size_t i = 0; i < crypto_scalarmult_curve25519_BYTES; ++i) {
           std::printf("%02x", remote_public_key[i]);
        }
        std::cout << std::endl;
        send(cfd, local_public_key, crypto_scalarmult_curve25519_BYTES,0);   //发送公钥数据
    }
    else if(len  == 0)
    {
         printf("Bob端断开了连接...\n");
    }
     else { perror("read"); }

    //计算共享密钥
    if(curve25519_donna(shared_secret1,local_private_key,remote_public_key)!=0){
       std::cerr << "计算本地公钥失败" << std::endl;
       return -1;
    }
    // 打印共享密钥
    std::cout << "共享密钥: ";
    for (size_t i = 0; i < crypto_scalarmult_curve25519_BYTES; ++i) {
         std::printf("%02x", shared_secret1[i]);
    }
    std::cout << std::endl;
    //接收客户端数据
    len = recv(cfd, shared_secret2, sizeof(shared_secret2),0);
    if(len > 0)
    {
        //比较两个共享密钥是否相同
        if (memcmp(shared_secret1, shared_secret2, crypto_scalarmult_curve25519_BYTES)!= 0) {
             std::cerr << "共享密钥不匹配。" << std::endl;
             return -1;
         }else {
            std::cout<<"共享密钥匹配"<<std::endl;
         }
        send(cfd, shared_secret1, crypto_scalarmult_curve25519_BYTES,0); //发送共享密钥
    }
    else if(len  == 0)
    {
         printf("Bob端断开了连接...\n");
    }
     else { perror("read"); }

     //随机生成一个nonce
     uint8_t nonce[crypto_box_NONCEBYTES];
     randombytes_buf(nonce, sizeof(crypto_box_NONCEBYTES));

     unsigned char message[MESSAGE_LEN] = "hello Welcome to DPC++!";  //加密信息
     unsigned char cipher_text[MESSAGE_LEN + crypto_box_MACBYTES];   //储存加密后的信息		
     std::cout << "加密的message: " << message << std::endl;
     
     if(crypto_box_easy_afternm(cipher_text, message, sizeof(message), nonce, shared_secret1)!=0){
        std::cerr << "加密信息失败。" << std::endl;
        return -1;
     }
     //发送数据
     send(cfd, cipher_text,MESSAGE_LEN + crypto_box_MACBYTES,0);
     send(cfd, nonce,crypto_box_NONCEBYTES,0);
     
     std::cout << "加密后的message: ";
     for (int i = 0; i < sizeof(cipher_text); i++) {
       std::cout << std::hex << static_cast<int>(cipher_text[i]);
     }
     std::cout << std::endl;
    //关闭套接字 
    close(cfd);
    close(lfd);
    return 0;
}
