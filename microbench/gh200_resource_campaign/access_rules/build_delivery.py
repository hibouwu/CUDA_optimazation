#!/usr/bin/env python3
"""Build the offline GH200 teaching HTML from reviewed Markdown, plots and validation data."""
import argparse,base64,hashlib,html,json,re
from pathlib import Path
import mistune

REPO=Path(__file__).resolve().parents[3]
DOC=REPO/'Docs/ModelEvaluation/gemm/experiments/gh200_sm90/access_rules'


def main():
    p=argparse.ArgumentParser();p.add_argument('--validation',required=True,type=Path);a=p.parse_args()
    validation=json.loads(a.validation.read_text());source=DOC/'RULES.md';text=source.read_text();assets={}
    def embed(match):
        label,path=match.groups();file=(DOC/path).resolve();data=file.read_bytes();assets[str(file.relative_to(REPO))]=hashlib.sha256(data).hexdigest()
        return f'![{label}](data:image/png;base64,{base64.b64encode(data).decode()})'
    text=re.sub(r'!\[([^]]*)\]\(([^)]+\.png)\)',embed,text)
    renderer=mistune.create_markdown(plugins=['table']);body=renderer(text)
    headings=[]
    def anchor(m):
        label=m[1];ident=f'section-{len(headings)+1}';headings.append((ident,re.sub('<[^>]+>','',label)));return f'<h2 id="{ident}">{label}</h2>'
    body=re.sub(r'<h2>(.*?)</h2>',anchor,body)
    body=re.sub(r'<h1>.*?</h1>','',body,count=1)
    charts=[('R11：消费者匹配后的联合服务','20261008-R11-job736989-v6/analysis/plots/r11-joint-service.png'),
            ('R12：独立区域的多路径竞争','20261008-R12-job736989-v3/analysis/plots/r12-paths.png'),
            ('R16：驻留与释放时序','20261008-R16-quota-job737122-v2/reanalysis/local-ten-processes/r16-services.png'),
            ('R18：完整时间与打点资格','20261008-R18-job737122-v1/reanalysis/formal-v1/r18-service.png'),
            ('R19：工作分配与CTA尾部','20261008-R19-job737122-v1/reanalysis/local-independent/r19-service.png'),
            ('B01：新增带宽与缓存坐标','20261008-B01-job737122-v4/reanalysis/local-independent/b01-services.png'),
            ('V07：完整时间与分项误差','20261008-V07-job737122-v1/reanalysis/validation-v2/v07-errors.png')]
    gallery=[]
    for title,path in charts:
        file=REPO/'results/gh200_resource_campaign/access_rules'/path;data=file.read_bytes();assets[str(file.relative_to(REPO))]=hashlib.sha256(data).hexdigest()
        gallery.append(f'<details><summary>{html.escape(title)}</summary><img loading="lazy" alt="{html.escape(title)}" src="data:image/png;base64,{base64.b64encode(data).decode()}"><p class="caption">点击图像放大，可切换原始分辨率。统计范围与条件见对应实验页。</p></details>')
    rows=[]
    for r in validation['cases']:
        rows.append('<tr>'+''.join(f'<td>{x}</td>' for x in [html.escape(r['case']),f"{r['predicted_us']:.3f}",f"{r['measured_us']:.3f}",f"{r['us_error']*100:+.2f}%",'合格' if r['trace_qualified'] else '仅plain'])+'</tr>')
    stats=validation['complete_time'];state='达标' if validation['complete_time_passed'] else '未整体达标'
    css='''*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f4f5f6;color:#182b39;font:16px/1.85 system-ui,"Noto Sans CJK SC",sans-serif}header{background:#163548;color:#fff;padding:50px max(24px,calc((100vw - 1180px)/2)) 38px}header h1{font-size:40px;line-height:1.3;margin:8px 0 18px;letter-spacing:.03em}.eyebrow{color:#8ccccc;font-size:13px;letter-spacing:.12em}.subtitle{color:#c8dce6;max-width:900px}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;max-width:1000px;margin-top:25px}.metric{border-top:2px solid #4e8297;padding-top:10px}.metric strong{display:block;font-size:28px}.metric span{font-size:13px;color:#c8dce6}.layout{display:grid;grid-template-columns:240px minmax(0,1fr);gap:34px;max-width:1230px;margin:35px auto;padding:0 24px}nav{position:sticky;top:20px;align-self:start;font-size:13px;line-height:1.6}nav a{display:block;padding:8px 10px;border-left:2px solid #d8e0e3;margin:3px 0}main{background:#fff;border:1px solid #e0e6e9;border-radius:10px;padding:28px 38px;min-width:0}h2{font-size:25px;margin:48px 0 18px;padding-top:14px;border-top:1px solid #d9e2e7;scroll-margin-top:20px}h3{font-size:20px}p{margin:12px 0 18px}a{color:#216d87;text-decoration:none}a:hover{text-decoration:underline}table{border-collapse:collapse;display:block;overflow-x:auto;font-size:14px;margin:22px 0;max-width:100%}th,td{padding:10px 13px;text-align:left;border-bottom:1px solid #dfe6ea}th{background:#edf3f6;white-space:nowrap}td:first-child{font-weight:550}code{background:#edf3f5;padding:2px 5px;border-radius:3px;font-size:.88em;overflow-wrap:anywhere}pre{background:#152f40;color:#e5eff4;border-radius:7px;padding:20px;overflow:auto;font:13px/1.7 ui-monospace,monospace}pre code{background:none;padding:0;color:inherit;white-space:pre;overflow-wrap:normal}img{display:block;width:100%;height:auto;margin:22px 0}details{border:1px solid #d9e4e9;border-radius:7px;margin:13px 0;padding:14px 18px}summary{cursor:pointer;font-weight:650}.notice{background:#fff4e5;border-left:4px solid #bc6b32;padding:18px 22px;border-radius:4px;margin-bottom:26px}.notice strong{display:block;font-size:19px}.caption,footer{font-size:12px;color:#5b6d78}footer{border-top:1px solid #dbe4e8;margin-top:40px;padding-top:22px}ul{padding-left:24px}.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:13px;color:#536975}.pill{border-radius:15px;background:#e7f1f1;color:#275765;padding:4px 12px}@media(max-width:900px){.layout{display:block}nav{position:static;display:flex;overflow:auto;gap:8px;margin-bottom:20px}nav a{min-width:180px}main{padding:22px}header h1{font-size:31px}.metrics{gap:12px}.metric strong{font-size:23px}}@media print{nav{display:none}.layout{display:block}header{padding:24px}main{border:none;padding:0}details{break-inside:avoid}a{color:inherit}}'''
    css+='dialog{width:96vw;max-width:96vw;border:0;border-radius:10px;padding:18px}dialog::backdrop{background:#102735c9}.zoom-head{display:flex;align-items:center;gap:15px;margin-bottom:12px}.zoom-head strong{flex:1}.zoom-scroll{overflow:auto;max-height:82vh}dialog img{width:100%;max-width:none;margin:0}dialog img.original{width:auto}button{font:inherit;cursor:pointer;border:1px solid #b5c8d1;background:#edf4f6;border-radius:6px;padding:5px 12px}main img{cursor:zoom-in}'
    zoom_ui='''<dialog id="zoom"><div class="zoom-head"><strong id="zoom-title"></strong><button id="zoom-size" type="button">原始分辨率</button><button id="zoom-close" type="button">关闭</button></div><div class="zoom-scroll"><img id="zoom-image" alt="实测图放大"></div></dialog><script>const viewer=document.getElementById('zoom'),picture=document.getElementById('zoom-image'),toggle=document.getElementById('zoom-size');document.querySelectorAll('main img').forEach(img=>{img.tabIndex=0;const show=()=>{picture.src=img.src;picture.classList.remove('original');toggle.textContent='原始分辨率';document.getElementById('zoom-title').textContent=img.alt;viewer.showModal()};img.addEventListener('click',show);img.addEventListener('keydown',e=>{if(e.key==='Enter')show()})});document.getElementById('zoom-close').addEventListener('click',()=>viewer.close());toggle.addEventListener('click',()=>{picture.classList.toggle('original');toggle.textContent=picture.classList.contains('original')?'适合宽度':'原始分辨率'});</script>'''
    nav=''.join(f'<a href="#{key}">{html.escape(label)}</a>' for key,label in headings)
    source_hash=hashlib.sha256(source.read_bytes()).hexdigest();validation_hash=hashlib.sha256(a.validation.read_bytes()).hexdigest()
    page=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GH200 条件规则与新留出验证</title><style>{css}</style></head><body>
<header><div class="eyebrow">GH200 · SM90a · CUTLASS 3.9.2 · 2026-10-08</div><h1>GH200 条件规则</h1><div class="subtitle">从寄存器源供给、共享内存和缓冲退役，到真实 tile 分配、关键 CTA 与完整 GEMM。数值随其对照、计时边界和适用范围使用。</div><div class="metrics"><div class="metric"><strong>124</strong><span>新增代表条件，控制与诊断另计</span></div><div class="metric"><strong>24</strong><span>带宽与缓存代表坐标</span></div><div class="metric"><strong>24</strong><span>先冻结预测的新留出条件</span></div></div></header>
<div class="layout"><nav aria-label="章节">{nav}<a href="#validation">V07 全部误差</a><a href="#figures">实测图</a></nav><main>
<div class="notice"><strong>V07 完整时间：{state}</strong>绝对误差中位数 {stats['median']*100:.2f}%，最大 {stats['maximum']*100:.2f}%；目标为 5% / 10%。初始供给、末次输出窗口与关键CTA定位尚未取得完整分项资格；短窗口另列，详见 <a href="V07-rule-validation.md">验证记录</a>。</div>
<div class="legend"><span class="pill">条件服务 ≠ 物理端口规格</span><span class="pill">数值正确 ≠ 预测达标</span><span class="pill">plain 与 trace 分别授予资格</span></div>
{body}<h2 id="validation">V07：全部 24 个新条件</h2><p>下表保留所有有效条件；误差为（预测−实测）/实测，使用冻结预测，不用测后实测频率回填。</p><table><thead><tr><th>条件</th><th>预测 μs</th><th>实测 μs</th><th>误差</th><th>trace</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2 id="figures">实测图</h2>{''.join(gallery)}
<footer>正文来自 RULES.md，SHA256 {source_hash}。验证数据 SHA256 {validation_hash}。图片已嵌入，可离线阅读；原始证据链接按仓库相对路径解析。完整可复算数据、失败记录和边界见各实验页。</footer></main></div>{zoom_ui}</body></html>'''
    output=DOC/'GH200-Rules-Analysis.html';output.write_text(page)
    (DOC/'GH200-Rules-Analysis.sources.json').write_text(json.dumps(dict(markdown=str(source.relative_to(REPO)),markdown_sha256=source_hash,validation=str(a.validation.resolve().relative_to(REPO)),validation_sha256=validation_hash,assets=assets,html_sha256=hashlib.sha256(output.read_bytes()).hexdigest()),indent=2)+'\n')
    print(output)
if __name__=='__main__':main()
