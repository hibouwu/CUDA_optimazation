"""cfg_b G3 开发候选：两域缓存与在途填充，叠加每 GPC 请求服务。"""
from collections import OrderedDict, defaultdict

from tma_gpc_service import GroupedTensorSource


class StatefulTensorSource:
    version = 'two-domain-cache-source/1'

    def __init__(self, work, cta_groups, prices, *, cache_bytes, hit_ns, home_ns, link_ns,
                 hbm_service_ns, trace_stores=True, ways=0, b_base_bytes=22*1024*1024, home_phase=0,
                 addresses=None, record_diagnostics=True, external_memory_service=None):
        self.coordinates = work
        self.cta_groups = cta_groups
        self.domains = [0 if g in (0, 4, 5, 6) else 1 for g in cta_groups]
        self.prices = prices
        self.b_base_bytes = b_base_bytes
        self.addresses = addresses or dict(A=0, B=b_base_bytes, records=96*1024*1024,
            ctas=128*1024*1024, worklist=130*1024*1024, gate=132*1024*1024)
        self.home_phase = home_phase
        self.hit_ns, self.home_ns, self.link_ns = hit_ns, home_ns, link_ns
        self.frontend_ns = prices[1]-hit_ns
        self.hbm_service_ns = hbm_service_ns
        self.external_memory_service = external_memory_service
        self.ways = ways
        self.capacity = ways or cache_bytes//128
        self.set_count = cache_bytes//128//ways if ways else 0
        self.caches = [defaultdict(OrderedDict), defaultdict(OrderedDict)] if ways else [OrderedDict(), OrderedDict()]
        self.hbm_pending = {}
        self.hbm_frontier = 0.
        self.trace_stores = trace_stores
        self.record_diagnostics = record_diagnostics
        self.line_sets = {}
        self.base = 0.
        self.work_index = 0
        self.hbm_lines = self.remote_lines = self.local_lines = self.pending_lines = 0
        self.diagnostics = []

    def home(self, line):
        # 专用分配推得的低 26 bit 候选；目标分配的常数分别测试。
        return (sum((line >> bit) & 1 for bit in (5, 6, 8, 10, 12, 14, 15, 17)) % 2) ^ self.home_phase

    def cache_set(self, domain, line):
        if not self.ways:
            return self.caches[domain]
        index = ((line & 31) | ((line >> 6) << 5)) % self.set_count
        return self.caches[domain][index]

    def put(self, domain, line, ready):
        cache = self.cache_set(domain, line)
        cache[line] = ready
        cache.move_to_end(line)
        if len(cache) > self.capacity:
            cache.popitem(last=False)

    def read(self, t, domain, line):
        cache = self.cache_set(domain, line)
        if line in cache:
            ready = cache[line]
            cache.move_to_end(line)
            if ready > t:
                self.pending_lines += 1
                return ready
            self.local_lines += 1
            return t+self.hit_ns
        home = self.home(line)
        cached = self.cache_set(home, line).get(line) if home != domain else None
        if cached is not None:
            self.cache_set(home, line).move_to_end(line)
            ready = max(cached, t+self.hit_ns)+self.link_ns
            self.remote_lines += 1
        else:
            previous = self.hbm_pending.get(line, 0.)
            if previous > t:
                home_ready = previous
                self.pending_lines += 1
            else:
                if self.external_memory_service is None:
                    self.hbm_frontier = max(t, self.hbm_frontier)+self.hbm_service_ns/128
                else:
                    request = self.external_memory_service.submit(t, read_bytes=128)
                    self.hbm_frontier = request['service_end_ns']
                home_ready = self.hbm_frontier+self.home_ns
                self.hbm_pending[line] = home_ready
                self.hbm_lines += 1
            self.put(home, line, home_ready)
            ready = home_ready+(self.link_ns if home != domain else 0.)
        self.put(domain, line, ready)
        return ready

    def store(self, t, domain, first_byte, length):
        for line in range(first_byte//128, (first_byte+length-1)//128+1):
            self.put(self.home(line), line, t)
            if domain is not None:
                self.put(domain, line, t)

    def begin(self, work_index, base, groups=64, probe_metadata=True):
        self.work_index, self.base = work_index, base
        self.groups = groups
        self.core = GroupedTensorSource(self.cta_groups, 2, self.prices[0], self.prices[1])
        self.core_pending = {}
        self.count = 0
        self.work = 0.
        self.busy = 0.
        self.diagnostics = []
        if not probe_metadata:
            return
        # CUDA memset 和 load_init 都在共同计时入口之前完成。
        self.store(base, None, self.addresses['ctas'], len(self.coordinates)*56)
        for cta, domain in enumerate(self.domains):
            first = self.addresses['worklist']+(work_index*len(self.coordinates)+cta)*8
            self.store(base, domain, first, 8)
        for domain in (0, 1):
            self.store(base, domain, self.addresses['gate'], 8)

    def memory_observer(self, t, cta, group, section):
        domain, t = self.domains[cta], t+self.base
        if section == 'cta_record':
            self.store(t, domain, self.addresses['ctas']+cta*56, 56)
        elif self.trace_stores:
            first = self.addresses['records']+(cta*self.groups+group)*80
            start, length = (0, 32) if section == 'issue_record' else (32, 48)
            self.store(t, domain, first+start, length)

    def submit(self, t, cta, key, weight, partition_demand=None):
        _, k, atom = key
        m, n = self.coordinates[cta][self.work_index]
        identity = ('A', m, k) if atom == 'A' else ('B', n*2+(atom == 'B1'), k)
        if identity not in self.line_sets:
            if atom == 'A':
                lines = [(self.addresses['A']+(m*128+row)*8192+k*128)//128 for row in range(128)]
            else:
                lines = [(self.addresses['B']+(k*64+row)*3072+n*256+(128 if atom == 'B1' else 0))//128
                         for row in range(64)]
            self.line_sets[identity] = lines
        t += self.base
        core_key = (self.cta_groups[cta], identity)
        previous = self.core_pending.get(core_key)
        merged = previous is not None and t < previous['service_end']
        core = previous if merged else self.core.submit(t, cta, identity, weight)
        self.core_pending[core_key] = core
        before = self.hbm_lines
        data_ready = max(self.read(t, self.domains[cta], line) for line in self.line_sets[identity])
        ready = max(core['source_ready'], data_ready+self.frontend_ns)
        self.count, self.work, self.busy = self.core.count, self.core.work, self.core.busy
        if self.record_diagnostics:
            self.diagnostics.append(dict(cta=cta,k=k,atom=atom,issue_ns=t-self.base,
                ready_ns=ready-self.base,new_hbm_lines=self.hbm_lines-before))
        return dict(source_id=core['source_id'],merged=merged,source_key=identity,
            service_start=None if merged else core['service_start']-self.base,
            service_end=None if merged else core['service_end']-self.base,
            source_ready=ready-self.base)


class G3GemmSource(StatefulTensorSource):
    """G3 两轮工作与输入服务共用缓存；在输出源读完成时登记 D 写入。"""

    def __init__(self,*args,output_base=64*1024*1024,**kwargs):
        super().__init__(*args,**kwargs)
        self.output_base=output_base
        self.output_bytes=0

    def begin(self,work_index,base,groups=64):
        super().begin(work_index,base,groups,probe_metadata=False)
        self.output_bytes=0

    def input(self,t,cta,group,atom,weight):
        self.work_index,k=divmod(group,64)
        busy=self.busy
        result=self.submit(t,cta,(cta,k,('A','B0','B1')[atom]),weight)
        result['source_busy_ns']=self.busy-busy
        return result

    def output(self,t,cta,subtile):
        output,j=divmod(subtile,8)
        m,n=self.coordinates[cta][output]
        # CUTLASS store: epi_n 外层、epi_m 内层；cfg_b 为 2×4 个 64×32 FP32 子块。
        first_row=m*128+(j%2)*64
        column=n*128+(j//2)*32
        for row in range(first_row,first_row+64):
            self.store(t+self.base,self.domains[cta],self.output_base+(row*1536+column)*4,128)
        self.output_bytes+=8192


class ProgramGemmSource(StatefulTensorSource):
    """完整cfg_b tile的共同程序接线；地址基址与CTA分组由调用方声明。"""

    def __init__(self, program, cta_groups, prices, *, output_base, **parameters):
        super().__init__(program['work'], cta_groups, prices, **parameters)
        self.ktiles = program['ktiles']
        self.output_base = output_base
        self.output_stride = program['case']['ldd']*4
        self.atom_templates = []
        for outputs in program['outputs']:
            templates = []
            for output in outputs:
                ids = output['operands']['A']['source_atoms'] + output['operands']['B']['source_atoms']
                templates.append([program['source_atoms'][i] for i in ids])
            self.atom_templates.append(templates)
        self.output_bytes = 0

    def begin(self, work_index, base, groups=64):
        super().begin(work_index, base, groups, probe_metadata=False)
        self.output_bytes = 0

    def input(self, t, cta, group, atom, weight):
        self.work_index, k = divmod(group, self.ktiles)
        operand = ('A', 'B0', 'B1')[atom]
        m, n = self.coordinates[cta][self.work_index]
        identity = ('A', m, k) if atom == 0 else ('B', n*2+(atom == 2), k)
        if identity not in self.line_sets:
            source = self.atom_templates[cta][self.work_index][atom]
            origin = [source['origin'][0], source['origin'][1]+64*k]
            strides, shape = source['stride_bytes'], source['shape']
            row_axis = 0 if atom == 0 else 1
            first = self.addresses[source['operand']]+source['base_offset_bytes']+sum(a*b for a,b in zip(origin,strides))
            row_bytes = shape[1-row_axis]*2
            lines = []
            for row in range(shape[row_axis]):
                start = first+row*strides[row_axis]
                lines.extend(range(start//128,(start+row_bytes-1)//128+1))
            self.line_sets[identity] = lines
        busy = self.busy
        result = self.submit(t,cta,(cta,k,operand),weight)
        result['source_busy_ns'] = self.busy-busy
        return result

    def output(self, t, cta, subtile):
        output, j = divmod(subtile,8)
        m,n = self.coordinates[cta][output]
        first_row = m*128+(j%2)*64
        column = n*128+(j//2)*32
        for row in range(first_row,first_row+64):
            self.store(t+self.base,self.domains[cta],self.output_base+row*self.output_stride+column*4,128)
        self.output_bytes += 8192
