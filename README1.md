
# 服务器使用说明

管理员：赵生云  
更新时间：2025-04-07

为保证服务器稳定、高效运行，请所有用户遵守以下使用规范与权限规则。

---
## 🧱 基础配置

| 项目  | 说明                           |
| --- | ---------------------------- |
| 系统  | Ubuntu 22.04                 |
| GPU | 2× RTX 4090 48GB             |
| 内存  | 128 GB                       |
| 系统盘 | 2 TB                         |
| 数据盘 | 2× 8 TB，挂载于 `/media/data`, `/media/disk` |
| IP | 173.17.135.76 |
| 主机名 | `zglab`|
| nvidia driver | 570|
| CUDA |11.8(默认)，12.1，12.8|

---

## 目录

1. [用户说明](#用户说明)
2. [登录方式](#登录方式)
3. [软件使用与安装说明](#软件使用与安装说明)
4. [磁盘使用说明](#磁盘使用说明)
5. [文件传输](#文件传输)
6. [网络连接说明](#网络连接说明)


---

## 用户说明

每位使用者拥有一个**独立的普通用户账户**，用于登录服务器、使用软件和保存个人数据。

- 当前已分配用户名：

```

renhe, chenzl, pangck, mengbo, zhaosy

```

- 初始密码统一为：

```

buaa2025

````
  - `ls`为管理员用户，请勿登录
### 🔄 修改密码方式

在终端执行：

```bash
passwd
````

按提示输入当前密码、新密码并确认。

### 📦 普通用户权限说明

|权限|是否可用|说明|
|---|---|---|
|访问自己主目录|✅|完全可读写，仅自己访问|
|使用常见软件|✅|Python、Conda、VSCode、Jupyter、CUDA 等|
|安装 Python 包|✅|建议使用 Conda 虚拟环境|
|创建 Conda 虚拟环境|✅|推荐每个项目使用独立环境|
|使用 `sudo`|❌|禁止普通用户使用 sudo|
|修改他人文件 / 杀他人进程|❌|系统禁止此类操作，违规将被记录|

#### 文件权限

- `/home/用户名/`：*仅本人可访问*
    
- `/media/data/public/`：公共读写目录

- `/media/data/projects/`：公共读写目录，协作项目目录
    
- `/meida/data/users/用户名/`：各用户独立数据目录，*仅本人可访问*
    

---

## 登录方式

### SSH 登录

```bash
ssh yourname@172.17.135.76
```

- 地址：`172.17.135.76`
    
- 端口：默认 `22`
    
- 支持用户名+密码登录
    
- 配置 SSH 密钥，如有需求请联系管理员
    
### vscode remote SSH 登录

- 由于之前的服务器中病毒可能是由vscode-server引入的，所有用户在使用vscode连接新服务器前，**务必重新安装本地的vscode**

### 远程桌面登录（多用户图形界面）

- Windows 用户可使用“远程桌面连接”
    
- 地址：`172.17.135.76`
    
- 每位用户**桌面环境独立**，互不干扰，可同时登录
    
- 若首次登录黑屏，请联系管理员初始化图形环境
    
---

## 软件使用与安装说明

### 普通用户

- ✅ 可使用已安装的软件：Python、CUDA、VSCode 等
    
- ✅ 可通过 `conda create` 创建虚拟环境
    
- ❌ 禁止使用 `sudo` 安装系统级软件
    
    

#### 推荐安装方式：

```bash
conda create -n myenv python=3.10
conda activate myenv
conda install numpy pandas
```
### CUDA使用说明

所有 CUDA Toolkit 安装在：

```
/usr/local/
```

包括以下目录：

```
/usr/local/cuda-11.8
/usr/local/cuda-12.1
/usr/local/cuda-12.8
```

默认软链接 `/usr/local/cuda` 通常指向当前系统默认 CUDA 版本（如 11.8）：

```bash
ls -l /usr/local/cuda
# 结果示例：cuda -> cuda-11.8
```


管理员已在 `/etc/profile.d/cuda-default.sh` 设置了系统默认 CUDA 版本（如 11.7）：

```bash
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

#### ✅ 如何切换其他版本 CUDA（每次登录生效）

服务器提供一个命令：`use-cuda`，用于快速切换不同版本。

##### 📥 使用方法：

```bash
source use-cuda <版本号>
```

##### 📌 示例：

```bash
source use-cuda 12.1     # 切换为 CUDA 12.1
source use-cuda 11.8     # 切回 CUDA 11.7
```

系统将自动设置对应环境变量，并验证版本：

```bash
nvcc -V
```

输出示例：

```
Cuda compilation tools, release 12.1, V12.1.105
```

---

#### ✅ 永久设置某版本（用户个性化）

如你希望每次登录默认使用 CUDA 12.1，可在你的 `~/.bashrc` 中添加：

```bash
source /usr/local/bin/use-cuda 12.1
```

保存后执行：

```bash
source ~/.bashrc
```

---

#### ✅ 相关命令参考

|命令|功能|
|---|---|
|`which nvcc`|查看当前使用的 nvcc 路径|
|`nvcc -V`|查看当前 CUDA 版本|
|`ls /usr/local/cuda-*`|查看系统安装的所有 CUDA 版本|
|`source use-cuda <version>`|切换 CUDA 环境变量|

---

#### 🚨 注意事项

- **不要使用 `sudo apt install nvidia-cuda-toolkit`** 安装 CUDA，它会安装版本较旧且容易破坏现有多版本配置。
    
- 所有 CUDA 版本均由管理员安装并统一管理，普通用户不建议私自更改 `/usr/local/` 下内容。
    
- 如需在 Docker / Conda 环境中使用特定 CUDA，请使用容器内置或自定义环境变量。
    

---

### 管理员权限

如需安装系统级依赖（如 apt 软件包、驱动等），请联系管理员申请。

---


## 磁盘使用说明

本服务器采用系统盘 + 数据盘分离结构，请严格区分两者用途：

### 📁 系统盘 `/`（2TB）

- 用于：系统运行、公共软件、用户主目录
    
- 禁止用于：保存大文件、数据集、模型、日志等
    
- 请定期清理 Conda 环境、pip 缓存等临时数据
    

### 📁 数据盘 `/media/data`, `/media/disk`（8TB × 2）

- 用于：存放数据集、模型、中间结果、输出日志
- 优先使用`/media/data`
- 图形桌面中，在`files`的`other location`访问，`sda1`为`/media/data`, `sdb1`为`/media/disk`
- 目录结构：

```
/media/data/
├── public/              # 公共协作目录
├── users/yourname/      # 用户个人数据
├── projects/xxx/        # 项目组空间（按需分配）
```

### ❄️ 冷数据规范

- 冷数据 = 长期保存、很少访问的数据
    
- 应移入：
    
```
/media/data/users/yourname/archive/
或
/media/data/public/归档/
```

### 🚫 禁止事项

- 不得将大文件保存在 `/home`
    
- 不得删除 `/meida/data/public/` 下非本人创建的数据
    
- 不得滥用 `/tmp` 存储（会被定期清理）
    

---

### 🔗 软连接使用说明


软连接是对原始文件/目录的引用，不占用额外存储空间，方便查看数据集

示例：

```bash
ln -s /meida/data/datasets/imagenet ~/imagenet
```

此命令会在 `~/imagenet` 创建一个指向 `/data/datasets/imagenet` 的软连接。

#### 🛠️ 常用命令

|操作|命令|
|---|---|
|创建软连接|`ln -s /真实路径 /链接名称`|
|查看软连接指向|`ls -l`|
|删除软连接（不影响原文件）|`rm 链接名`|
|替换已有链接|`rm 链接名 && ln -s 新目标`|

> ✅ 推荐使用绝对路径进行链接，确保跨目录有效。

---

## 文件传输

### 1. 命令行方式（推荐）

```bash
scp yourfile.txt yourname@172.17.135.76:/home/yourname/
scp -r yourdir/ yourname@172.17.135.76:/home/yourname/
```

- 网速可达 80MB/s

### 2. 图形方式

使用 `FileZilla`、`WinSCP`、Remmina 等支持 SFTP 协议的客户端连接。

### 3. 数据迁移

## 网络连接说明

### 🔐 网络认证（自动登录机制）

服务器已部署**认证守护进程**，用于应对校园网的门户登录限制：

- 开机后系统会自动完成联网认证，无需人工干预
- 如果连接异常（如 `ping baidu.com` 无响应），请执行以下命令手动尝试认证：

```bash
systemctl restart auth-client
```

如仍无法联网，请联系管理员协助。

---

### 🌍 配置网络代理（用户自选）
```bash
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export all_proxy=socks5://127.0.0.1:7890
```

部分用户如需访问国外资源（如 GitHub、PyPI、Hugging Face），可**自行配置本地代理客户端**（如 Clash、V2Ray、Shadowsocks 等）并设置用户环境变量。
**不要设置全系统的环境变量**



## 禁止事项（使用行为规范）

- 不擅自占用全部 GPU
    
- 不运行无日志、长期占资源的后台程序
    
- 不破坏他人环境、删除系统关键路径
    
- 所有任务建议通过日志记录便于调试排查
    
- 不允许用户互相使用或共享账号
    

---

## 👨‍💻 管理员

如需以下支持请联系管理员：

- 安装软件 / 系统级依赖
    
- 设置共享目录或项目组权限
    
- 开通远程桌面 / sudo 权限（临时）
    
- 协调 GPU / CPU 资源使用
    

**感谢配合，祝大家科研顺利！🚀**
