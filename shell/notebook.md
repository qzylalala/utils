常用 shell 指令
=================

## 数组

bash只支持一维数组，不支持多维数组

* 定义数组：array_name=(li wang xiang zhang) （小括号做边界、使用空格分离）
* 单独定义数组的元素： array_para[0]="w"; array_para[3]="s" （定义时下标不连续也可以）
* 赋值数组元素：array_name[0]="zhao";
* 获取数组元素：echo ${array_name[@]} # 输出"li zhang" 输出数组所有元素，没有元素的下标省略
* 取得元素个数：\${#array_name[@]} 或者 \${#array_name}
* 取得单个元素长度：\${#array_name[1]}

## 参数传递

获取参数值：
* $0 ： 固定，代表执行的文件名
* $1 ： 代表传入的第1个参数
* $n ： 代表传入的第n个参数
* \$#：参数个数
* \$\*： 以一个单字符串显示所有向脚本传递的参数。如"$*"用 ["] 括起来的情况、以"$1 $2 … $n"的形式输出所有参数
* \$@：与 $*相同，但是使用时加引号，并在引号中返回每个参数。
* \$\$：脚本运行的当前进程号
* \$!：后台运行的最后一个进程的ID
* \$?： 显示最后命令的退出状态。0表示没有错误，其他任何值表明有错误。
* \$* 与 \$@ 区别
    * 相同点：都是引用所有参数。
    * 不同点：只有在双引号中体现出来。假设在脚本运行时写了三个参数 1、2、3，，则 " * " 等价于 "1 2 3"（传递了一个参数），而 "@" 等价于 "1" "2" "3"（传递了三个参数）。