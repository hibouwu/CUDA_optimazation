# S14 压缩证据只读校验事务增量草案

状态：待未实施者独立审查；不修改已审resolver，不授GPU或桥接资格。范围只覆盖S14有限历史policy的大量metadata/hash读取。

`PackedEvidence.file_sha256/read_json` 当前每次核对实际archive SHA。原84的675个明确review context有数万门禁查找，一份348,887,480 B包重复hash会造成TB级读取。这里保留实际字节身份边界，只将重复核对移到一段明确的只读校验事务首尾。

## 最小接口

新增 `verified_transaction(required_small_objects)` context manager。进入前必须满足：整个 `verify_all(small_objects=..., observer=...)` 成功返回；所需metadata均在受限16/64MiB白名单缓存中；再次核对实际archive/index字节SHA与长度。失败不能yield只读view。

view仅有 `read_bytes/read_json/file_sha256/inventory`，路径仍采用已核验完整逻辑member namespace。read只准访问声明并缓存的小对象，不允许回落为另一次隐含xz扫描或任意新payload读取。file_sha与inventory只返回本次完整verified扫描的成员表副本。事务中不执行历史代码、GPU、编译或外部模块派发；仅在代码拥有的S14有限schema/旧review context/计数/资源/identity核对中使用。

退出时再次核对实际archive/index字节SHA。无异常且末身份仍一致才产生 `transaction_receipt`，包含pack/index SHA、所用member白名单和closed=true。任意checker异常、身份漂移或未完整退出使事务失败，不输出资格；view退出后失效，不能成为可跨事务沿用的隐式缓存。保留原r3方法的逐次身份检查行为，旧S12/S13不进入新接口。

## 外部checker的发布顺序

1. 固定closure/index/原review/coverage身份先核对；准备代码拥有的固定84 artifact描述，descriptor及metadata都是原被审字节的派生数据。
2. 使用一次完整verify_all，observer逐chunk核对全部payload/guard和lifecycle words；helper只有整个包与所有成员通过后才publish缓存。observer不能授B3。
3. observer.finish核对所有预期artifact均完整消费、无残余word，重算总计2967828384个payload/guard与231192个lifecycle word。
4. 在verified_transaction内解析全部固定metadata及675个明确review context，生成暂存normindex。只能按原spec声明的snapshot/repo根解释门禁；initial历史JSON仅保留原hash，不猜live根。
5. 事务成功关闭后，才将全值/metadata/完整身份证明写入实施记录。桥接资格仍需未实施者独立B签署；unsigned normindex、事务receipt或observer pass均不自行授formal/原B3新资格。

## 独立验收

负例覆盖：未verify或缺白名单进入、事务中pack/index改变、checker异常、view关闭后访问、非白名单读取、旧schema/namespace或错误role；任何失败不publish成功receipt或derived qualification。使用实际原84包核对hash调用次数和读取字节量，事务首尾各一次，不能每gate一次；记录decoded bytes、峰值RSS与耗时。另一代理审查后由root实现，实施者不签自身B。

## r2：准备边界与失败终态

metadata descriptor只能来自第一次完整verify_all成功后的白名单缓存；先用一次只读事务解析这些已核字节，事务关闭成功后准备observer。随后第二次完整verify_all加observer.finish，再使用最终只读事务生成normindex。准备阶段与最终阶段各自记录事务身份，两个逻辑verify_all仍是四次完整decoder pass，不把descriptor准备放到未核验包之前。

事务进入时除了重算实际archive/index SHA，还逐项核对将提供给view的缓存长度和成员SHA。禁止nested事务，禁止active事务期间再次verify_all；非法操作使当前事务sticky失败。view的任一方法失败必须sticky标记；调用方捕获该异常也不能使事务成功关闭。checker异常只能记录并re-raise；独立审查要确认调用代码没有吞异常后发布closed=true。

receipt仅在context manager正常退出、sticky状态无失败、末身份再次核对一致且view已经失效之后才能读取。未关闭、失败、异常退出和关闭后的view均不能发布成功receipt或继续访问缓存。旧逐次核身份的方法仍保留；事务只用于经过固定代码拥有的schema/角色/identity检查。即使receipt成功，也不授任何桥接、B3或性能资格。
