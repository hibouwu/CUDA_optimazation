# 主线复核128线程判读

B的b80ac5e、d8e4f82分别合入5bc020b、108664d。旧四leader入口遇release_threads明确拒绝；v2接口核对全部256..383线程和原leader一致性，不静默丢弃其余124个记录。

主线独立重放740056四份原始gz，完整processes（包括每份128个线程字段）、协议绑定、配对门槛、条件区间与不合格原因均与B-release-thread-smoke-v1相同；仅input_sha256代码位置不同。40项release/decoder/sparse相关测试通过。Markdown按实际协议与metrics生成，不继承旧v1的固定失败句。

interpretation_qualified=false。单对的local差0/+1、上游差−6/+14、event−0.5414%/0%不等于十对资格；stage4[206,231]和stage6[210,232]cycle仍只作该未合格pilot的条件界，不采用机制或服务价格。
