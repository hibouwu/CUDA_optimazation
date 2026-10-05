"""Independent logical replay; no CUDA/runtime/host workload helper imports."""
from common.suite_io import require
FORMS=('local_read','dsm_read','local_write','dsm_write','cluster_sync','bulk_single_target','bulk_all_targets')
PAIRS=((1,0),(2,3),(5,4294967295))

def uint(x,bits=32):
 require(type(x)is int and 0<=x<2**bits,'S17 uint domain');return x

def dimensions(form,c,g,i,seed):
 require(form in FORMS and type(c)is int and c in (2,4,8) and type(g)is int and 0<g<=1024 and (i,seed) in PAIRS,'finite S17 short domain')
 return c*g

def mask(form,c):return (1<<c)-1 if form=='bulk_all_targets' else 1<<(c-1) if form=='bulk_single_target' else 0

def shapes(form,c,g,i):
 b=c*g;w=4096 if form.startswith('bulk_') else 1024 if form!='cluster_sync' else 0
 s={'completion':[b,6,2],'lifecycle':[b,i,12],'stamps':[b,5,2]}
 if w:s={'trace':[b,i,w],'guards':[b,8],**s}
 if form in FORMS[:4]:s['sums']=[b,128]
 if form.startswith('bulk_'):s['tokens']=[b,i,2];s['source_ring']=[g,32,4096]
 return s

def ledger(form,c,g,i):
 b=c*g;m=mask(form,c);receivers=m.bit_count()
 return {'request_count':g*i if m else b*i if form!='cluster_sync' else 0,'source_request_bytes':g*i*16384 if m else b*i*4096 if form!='cluster_sync' else 0,'received_bytes':g*i*receivers*16384 if m else 0,'collective_sync_stages':g*i*(2 if m or 'write' in form else 1),'participants_per_cluster':c,'opaque_token_words':2*g*receivers*i if m else 0}

def payload(form,c,block,item,word,seed):
 rank=block%c;base=block-rank
 if form.startswith('bulk_'):
  selected=bool(mask(form,c)&(1<<rank));v=(17*((block//c*32+(item%32 if selected else 0))*4096+word)+seed)&0xffffffff
  return v if selected else v^0xffffffff
 source=(rank+1)%c if form=='dsm_read' else (rank-1)%c if form=='dsm_write' else rank
 return ((29 if 'write' in form else 17)*((base+source)*1024+word)+seed+(31*item if 'write' in form else 0))&0xffffffff

def lifecycle(form,c,block,item):
 rank=block%c;m=mask(form,c);recv=bool(m&(1<<rank)) if m else form!='cluster_sync'
 return [rank,c,item,m,int(recv),int(bool(m) and recv),16384 if m and recv else 0,int(bool(m) and rank==0),int(bool(m) and recv),int(form!='cluster_sync'),1,1]

def audit(form,c,g,i,seed,a):
 b=dimensions(form,c,g,i,seed);layout=shapes(form,c,g,i)
 require(set(a)==set(layout),'exact S17 output component set')
 for key,shape in layout.items():
  count=1
  for n in shape:count*=n
  require(len(a[key])==count,'S17 full shape')
  for v in a[key]:uint(v)
 def u64(key,index):return a[key][2*index]|(a[key][2*index+1]<<32)
 for block in range(b):
  require([u64('completion',block*6+k) for k in range(6)]==[i,block%c,c,1,1,0],'S17 completed consumers and exit')
  for item in range(i):
   off=(block*i+item)*12;require(a['lifecycle'][off:off+12]==lifecycle(form,c,block,item),'S17 phase role/mask/arrival/acquire/consumer_done/release')
  ns0,ns1,cy0,cy1,smid=[u64('stamps',block*5+k) for k in range(5)]
  require(ns0<=ns1 and cy0<=cy1 and all(v!=2**64-1 for v in (ns0,ns1,cy0,cy1)) and smid<2**32-1,'S17 same CTA time domain; physical SMID need not be contiguous')
  if 'trace' in a:
   w=4096 if form.startswith('bulk_') else 1024
   for item in range(i):
    off=(block*i+item)*w;require(all(v==payload(form,c,block,item,word,seed) for word,v in enumerate(a['trace'][off:off+w])),'S17 full payload')
   require(a['guards'][block*8:block*8+8]==[0xd15ea5e0+k for k in range(8)],'S17 guard')
  if 'sums' in a:
   for thread in range(128):
    expected=sum(payload(form,c,block,item,thread+128*j,seed) for item in range(i) for j in range(8))&0xffffffff
    require(a['sums'][block*128+thread]==expected,'S17 consumed lane sum')
 if 'tokens' in a:
  for block in range(b):
   if not mask(form,c)&(1<<(block%c)):
    require(all(u64('tokens',block*i+item)==0 for item in range(i)),'S17 nonreceiver never arrives/waits; token field reset0')
 if 'source_ring' in a:
  for cluster in range(g):
   for slot in range(32):
    off=(cluster*32+slot)*4096
    require(all(v==(17*((cluster*32+slot)*4096+w)+seed)&0xffffffff for w,v in enumerate(a['source_ring'][off:off+4096])),'S17 complete unchanged source ring')
 # Selected opaque tokens are only uint64-domain evidence, excluded from exact count.
 return {'status':'pass','exact_checked_elements':sum(len(x) for x in a.values())-ledger(form,c,g,i)['opaque_token_words'],'opaque_token_words_preserved_domain_only':ledger(form,c,g,i)['opaque_token_words'],**ledger(form,c,g,i)}
